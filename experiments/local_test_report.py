"""Local GPU reevaluation and extended reports from frozen test predictions."""
import os
import platform
import time

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .final_test_package import FinalTest, activity, atomic_csv, atomic_json, digest, read_json


class LocalTest(FinalTest):
    def __init__(self, root, run_tag, batch_size=8):
        if not run_tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_tag):
            raise ValueError('RUN_TAG chỉ gồm chữ, số, gạch dưới hoặc gạch ngang.')
        if batch_size < 1:
            raise ValueError('Batch size phải dương.')
        super().__init__(root, overrides={
            'output_dir': f'final_test/local_runs/{run_tag}', 'eval_batch_size': batch_size})

    def prepare(self):
        if not self.cached() and not torch.cuda.is_available():
            raise RuntimeError('Kernel hiện tại không có CUDA. Chạy scripts/setup_vscode_gpu.ps1 và chọn kernel LDTF GPU.')
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        super().prepare()
        atomic_json(self.out / 'logs/environment.json', {
            'python': platform.python_version(), 'torch': str(torch.__version__),
            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'batch_size': self.cfg['eval_batch_size'], 'dtype': 'float32', 'timestamp': time.time()})

    def import_previous(self):
        """Import only predictions proven by the original artifact manifest."""
        if not self.checked:
            raise RuntimeError('Kiểm tra đầu vào và checkpoint trước.')
        with activity('Xác minh dự đoán Colab đã lưu'):
            previous = self.root / 'final_test/B2_bert_finetuned_cls_seed42'
            source = previous / 'predictions/test_predictions.npz'
            manifest = read_json(previous / 'reports/artifact_manifest.json')
            original_config = read_json(self.root / 'final_test/config.json')
            import hashlib, json, shutil
            identity = hashlib.sha256(json.dumps(original_config, sort_keys=True).encode()).hexdigest()
            if manifest['identity'] != identity:
                raise ValueError('Manifest Colab không khớp cấu hình đã chốt.')
            hashes = {k.replace('\\', '/'): v for k, v in manifest['files'].items()}
            if digest(source) != hashes['predictions/test_predictions.npz']:
                raise ValueError('Dự đoán Colab sai checksum.')
            if self.cached():
                return
            temporary = self.pred_path.with_suffix('.tmp')
            shutil.copyfile(source, temporary)
            os.replace(temporary, self.pred_path)
            atomic_json(self.state_path, {'identity': self.identity, 'status': 'predicted',
                'predictions_sha256': digest(self.pred_path), 'origin': 'imported_colab_predictions'})


def extended_report(workflow, bootstrap_samples=1000):
    """No access to the test loader; all analyses use saved predictions only."""
    if not workflow.cached():
        raise RuntimeError('Chưa có dự đoán hoàn chỉnh.')
    if bootstrap_samples < 100:
        raise ValueError('Dùng ít nhất 100 bootstrap samples.')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, f1_score
    out, names = workflow.out, workflow.cfg['label_names']
    with tqdm(total=6, desc='Báo cáo mở rộng', unit='bước') as progress:
        progress.set_postfix_str('Xác minh dự đoán và tính xác suất')
        with np.load(workflow.pred_path, allow_pickle=False) as data:
            logits, labels, texts = data['logits'], data['labels'], data['texts']
            rows = data['row_index']
        if logits.shape != (workflow.cfg['expected_test_rows'], 4) or not np.isfinite(logits).all():
            raise ValueError('Logits sai shape hoặc chứa NaN/Inf.')
        if labels.shape != (len(logits),) or not np.isin(labels, range(4)).all():
            raise ValueError('Nhãn không hợp lệ.')
        shifted = logits.astype(float) - logits.max(axis=1, keepdims=True)
        probs = np.exp(shifted); probs /= probs.sum(axis=1, keepdims=True)
        predicted = probs.argmax(1); confidence = probs.max(1); correct = predicted == labels
        table = pd.DataFrame({'row_index': rows, 'text': texts, 'true_label': np.asarray(names)[labels],
            'predicted_label': np.asarray(names)[predicted], 'confidence': confidence, 'correct': correct,
            'word_count': [len(t.split()) for t in texts]})
        for i, name in enumerate(names):
            table[f'p_{name}'] = probs[:, i]
        atomic_csv(table, out / 'predictions/all_predictions.csv', index=False)
        atomic_csv(table[~correct].sort_values('confidence', ascending=False), out / 'predictions/high_confidence_errors.csv', index=False)
        progress.update(1)
        progress.set_postfix_str('Ma trận chuẩn hóa và cặp nhãn hay nhầm')
        from sklearn.metrics import confusion_matrix
        matrix = confusion_matrix(labels, predicted, labels=range(4))
        normalized = matrix / np.maximum(matrix.sum(1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(8, 7))
        ConfusionMatrixDisplay(normalized, display_labels=names).plot(ax=ax, cmap='Blues', values_format='.1%', colorbar=False)
        ax.set_title('Tỷ lệ dự đoán theo nhãn thật'); fig.tight_layout()
        fig.savefig(out / 'figures/confusion_matrix_normalized.png', dpi=180); plt.close(fig)
        pairs = pd.DataFrame([{'true_label': names[i], 'predicted_label': names[j], 'count': int(matrix[i,j])}
            for i in range(4) for j in range(4) if i != j]).sort_values('count', ascending=False)
        atomic_csv(pairs, out / 'metrics/confusion_pairs.csv', index=False)
        progress.update(1)
        progress.set_postfix_str('Hiệu chuẩn xác suất: reliability và ECE')
        bins = np.minimum((confidence * 10).astype(int), 9)
        calibration = []
        for i in range(10):
            mask = bins == i
            calibration.append({'bin_lower': i/10, 'bin_upper': (i+1)/10, 'count': int(mask.sum()),
                'mean_confidence': float(confidence[mask].mean()) if mask.any() else None,
                'accuracy': float(correct[mask].mean()) if mask.any() else None})
        cal = pd.DataFrame(calibration)
        ece = float(((cal.mean_confidence-cal.accuracy).abs().fillna(0)*cal['count']).sum()/len(labels))
        brier = float(np.square(probs - np.eye(4)[labels]).sum(1).mean())
        atomic_csv(cal, out / 'metrics/calibration_bins.csv', index=False)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        populated = cal['count'] > 0
        axes[0].plot([0,1],[0,1], '--', color='gray')
        axes[0].plot(cal.loc[populated,'mean_confidence'], cal.loc[populated,'accuracy'], 'o-', color='#26786e')
        axes[0].set(xlabel='Xác suất dự đoán trung bình', ylabel='Tỷ lệ đúng', xlim=(0,1), ylim=(0,1), title=f'Reliability / ECE={ece:.4f}')
        axes[1].hist([confidence[correct], confidence[~correct]], bins=np.linspace(0,1,21), label=['Đúng','Sai'], color=['#26786e','#b55561'])
        axes[1].set(xlabel='Xác suất nhãn dự đoán', ylabel='Số mẫu'); axes[1].legend()
        fig.tight_layout(); fig.savefig(out / 'figures/calibration.png', dpi=180); plt.close(fig)
        progress.update(1)
        progress.set_postfix_str('Bootstrap khoảng tin cậy 95%')
        rng = np.random.default_rng(42); values = []
        with tqdm(range(bootstrap_samples), desc='Bootstrap', unit='lượt', leave=False) as bar:
            bar.set_postfix_str('Lấy mẫu lại dự đoán, không train')
            for _ in bar:
                idx = rng.integers(0, len(labels), size=len(labels))
                values.append((correct[idx].mean(), f1_score(labels[idx], predicted[idx], labels=range(4), average='macro', zero_division=0)))
        bounds = np.quantile(values, [0.025,0.975], axis=0)
        stats = {'ece_10_bins': ece, 'multiclass_brier_sum': brier, 'bootstrap_samples': bootstrap_samples,
            'bootstrap_seed': 42, 'accuracy_ci95': bounds[:,0].tolist(), 'macro_f1_ci95': bounds[:,1].tolist(),
            'correct': int(correct.sum()), 'incorrect': int((~correct).sum())}
        atomic_json(out / 'metrics/extended_metrics.json', stats)
        progress.update(1)
        progress.set_postfix_str('Thống kê theo độ dài văn bản')
        table['length_group'] = pd.cut(table.word_count, bins=[-1,25,50,100,float('inf')], labels=['0-25','26-50','51-100','101+'])
        length_stats = table.groupby('length_group', observed=False).agg(count=('correct','size'), accuracy=('correct','mean'))
        atomic_csv(length_stats, out / 'metrics/by_word_length.csv')
        progress.update(1)
        progress.set_postfix_str('Báo cáo HTML và Markdown')
        metrics = read_json(out / 'metrics/test_metrics.json')
        title = 'Báo cáo đánh giá B2 trên tập test'
        explanation = ('Đây là đánh giá lại checkpoint đã chốt, không phải thí nghiệm độc lập mới. '
            'Khoảng tin cậy bootstrap phản ánh biến động do lấy mẫu test, không đo độ ổn định qua seed huấn luyện. '
            'Xác suất softmax là mức tự tin của model, không phải bảo đảm đúng. '
            'ECE dùng 10 khoảng đều; Brier là tổng sai số bình phương trên bốn lớp. '
            'Phân tích độ dài dùng số từ tách bằng khoảng trắng, không phải token BERT. '
            'Test thuộc AG News; không dùng các lỗi test để chỉnh model rồi báo lại điểm trên cùng test.')
        summary = (f"Accuracy: {metrics['accuracy']:.6f}; Macro F1: {metrics['f1_macro']:.6f}; "
            f"đúng {stats['correct']}/{len(labels)}; sai {stats['incorrect']}. "
            f"Accuracy CI95: {bounds[0,0]:.6f}–{bounds[1,0]:.6f}; "
            f"Macro F1 CI95: {bounds[0,1]:.6f}–{bounds[1,1]:.6f}. "
            f"ECE: {ece:.6f}; Brier (tổng bốn lớp): {brier:.6f}.")
        per_class = pd.read_csv(out / 'metrics/per_class_metrics.csv')
        images = ['confusion_matrix.png','confusion_matrix_normalized.png','per_class_f1.png','calibration.png']
        body = f'<h1>{title}</h1><p>{summary}</p><p>{explanation}</p>'
        body += '<h2>Chỉ số từng nhãn</h2>' + per_class.to_html(index=False, float_format=lambda x: f'{x:.4f}')
        for name in images:
            body += f'<img src="../figures/{name}" alt="{name}">'
        body += '<h2>Cặp nhãn hay nhầm</h2>' + pairs.head(6).to_html(index=False)
        body += '<h2>Kết quả theo độ dài văn bản (số từ)</h2>' + length_stats.to_html()
        body += '<h2>20 mẫu sai tự tin nhất</h2>' + table[~correct].sort_values('confidence',ascending=False).head(20)[['row_index','text','true_label','predicted_label','confidence']].to_html(index=False, escape=True)
        document = '<!doctype html><html lang="vi"><meta charset="utf-8"><title>'+title+'</title><style>body{font:16px Arial;max-width:1100px;margin:32px auto;padding:16px;color:#202528}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border:1px solid #ccc;text-align:left;overflow-wrap:anywhere}img{max-width:100%;display:block;margin:24px auto}p{line-height:1.6}</style>'+body+'</html>'
        (out / 'reports/DETAILED_REPORT.html').write_text(document, encoding='utf-8')
        (out / 'reports/DETAILED_REPORT.md').write_text(f'# {title}\n\n{summary}\n\n{explanation}\n\n'+ '\n\n'.join(f'![{n}](../figures/{n})' for n in images), encoding='utf-8')
        progress.update(1); progress.set_postfix_str('Hoàn tất')
    return stats


def export_local(workflow):
    import zipfile
    with activity('Đóng gói báo cáo local'):
        files = [p for p in workflow.out.rglob('*') if p.is_file() and p.suffix not in ('.tmp','.lock')
                 and p.name != 'artifact_manifest.json']
        manifest = workflow.out / 'reports/artifact_manifest.json'
        atomic_json(manifest, {'identity': workflow.identity, 'files': {p.relative_to(workflow.out).as_posix(): digest(p) for p in files}})
        bundle = workflow.out.with_suffix('.zip')
        with zipfile.ZipFile(bundle.with_suffix('.tmp'), 'w', zipfile.ZIP_DEFLATED) as archive:
            for p in files+[manifest]:
                archive.write(p,p.relative_to(workflow.out).as_posix())
        os.replace(bundle.with_suffix('.tmp'),bundle)
    return bundle
