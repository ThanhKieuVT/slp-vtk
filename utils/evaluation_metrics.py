"""
Evaluation Metrics for Sign Language Generation
Includes: BLEU, ROUGE, DTW, Pose-specific metrics
"""
import numpy as np
import torch
from typing import List, Dict, Tuple
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


class SignLanguageEvaluator:
    """Comprehensive evaluator for sign language generation"""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1
        
    # ==================== TEXT METRICS ====================
    
    def compute_bleu(self, references: List[List[str]], hypotheses: List[List[str]]) -> Dict[str, float]:
        """
        Compute BLEU scores (BLEU-1 to BLEU-4)
        
        Args:
            references: List of reference sequences (tokenized), shape [N, ref_tokens]
            hypotheses: List of generated sequences (tokenized), shape [N, hyp_tokens]
        
        Returns:
            Dict with bleu-1, bleu-2, bleu-3, bleu-4, corpus-bleu
        """
        assert len(references) == len(hypotheses), "Mismatch in number of samples"
        
        bleu_scores = {
            'bleu-1': [],
            'bleu-2': [],
            'bleu-3': [],
            'bleu-4': []
        }
        
        # Sentence-level BLEU
        for ref_seq, hyp_seq in zip(references, hypotheses):
            # ref_seq should be list of lists for multiple references
            if not isinstance(ref_seq[0], list):
                ref_seq = [ref_seq]
            
            # BLEU-n scores
            for n in range(1, 5):
                weights = tuple([1.0/n] * n + [0.0] * (4-n))
                score = sentence_bleu(ref_seq, hyp_seq, weights=weights, smoothing_function=self.smoothing)
                bleu_scores[f'bleu-{n}'].append(score)
        
        # Corpus-level BLEU-4
        corpus_bleu_score = corpus_bleu(
            [[ref] if not isinstance(ref[0], list) else ref for ref in references],
            hypotheses,
            smoothing_function=self.smoothing
        )
        
        # Average
        result = {k: np.mean(v) for k, v in bleu_scores.items()}
        result['corpus-bleu'] = corpus_bleu_score
        
        return result
    
    def compute_rouge(self, references: List[str], hypotheses: List[str]) -> Dict[str, float]:
        """
        Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
        
        Args:
            references: List of reference strings
            hypotheses: List of generated strings
        
        Returns:
            Dict with rouge-1, rouge-2, rouge-l (F1 scores)
        """
        assert len(references) == len(hypotheses)
        
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for ref, hyp in zip(references, hypotheses):
            scores = self.rouge_scorer.score(ref, hyp)
            for key in rouge_scores.keys():
                rouge_scores[key].append(scores[key].fmeasure)
        
        return {k: np.mean(v) for k, v in rouge_scores.items()}
    
    # ==================== POSE METRICS ====================
    
    def compute_dtw(self, gt_poses: np.ndarray, pred_poses: np.ndarray) -> float:
        """
        Compute DTW (Dynamic Time Warping) distance between pose sequences
        
        Args:
            gt_poses: Ground truth poses [T1, D]
            pred_poses: Predicted poses [T2, D]
        
        Returns:
            DTW distance (lower is better)
        """
        # Flatten each frame if multi-dimensional
        if gt_poses.ndim > 2:
            gt_poses = gt_poses.reshape(gt_poses.shape[0], -1)
        if pred_poses.ndim > 2:
            pred_poses = pred_poses.reshape(pred_poses.shape[0], -1)
        
        distance, _ = fastdtw(gt_poses, pred_poses, dist=euclidean)
        
        # Normalize by sequence length
        avg_len = (len(gt_poses) + len(pred_poses)) / 2
        return distance / avg_len
    
    def compute_pose_mse(self, gt_poses: np.ndarray, pred_poses: np.ndarray, 
                         align_length: bool = True) -> float:
        """
        Compute Mean Squared Error between pose sequences
        
        Args:
            gt_poses: Ground truth [T, D] or [T, J, 2]
            pred_poses: Predicted [T, D] or [T, J, 2]
            align_length: If True, truncate to min length
        
        Returns:
            MSE value
        """
        if align_length:
            min_len = min(len(gt_poses), len(pred_poses))
            gt_poses = gt_poses[:min_len]
            pred_poses = pred_poses[:min_len]
        
        assert gt_poses.shape == pred_poses.shape, "Shape mismatch after alignment"
        
        return np.mean((gt_poses - pred_poses) ** 2)
    
    def compute_acceleration_error(self, gt_poses: np.ndarray, pred_poses: np.ndarray) -> float:
        """
        Compute acceleration error (smoothness metric)
        
        Args:
            gt_poses: [T, D]
            pred_poses: [T, D]
        
        Returns:
            Mean acceleration error
        """
        min_len = min(len(gt_poses), len(pred_poses))
        gt_poses = gt_poses[:min_len]
        pred_poses = pred_poses[:min_len]
        
        # Compute second-order differences (acceleration)
        if len(gt_poses) < 3:
            return 0.0
        
        gt_accel = gt_poses[2:] - 2 * gt_poses[1:-1] + gt_poses[:-2]
        pred_accel = pred_poses[2:] - 2 * pred_poses[1:-1] + pred_poses[:-2]
        
        return np.mean((gt_accel - pred_accel) ** 2)
    
    # ==================== BATCH EVALUATION ====================
    
    def evaluate_batch(self, 
                      gt_poses: List[np.ndarray],
                      pred_poses: List[np.ndarray],
                      gt_texts: List[str] = None,
                      pred_texts: List[str] = None,
                      gt_glosses: List[List[str]] = None,
                      pred_glosses: List[List[str]] = None) -> Dict[str, float]:
        """
        Comprehensive evaluation on a batch
        
        Args:
            gt_poses: List of ground truth pose sequences
            pred_poses: List of predicted pose sequences
            gt_texts: List of ground truth text strings (for ROUGE)
            pred_texts: List of predicted text strings (for ROUGE)
            gt_glosses: List of ground truth gloss tokens (for BLEU)
            pred_glosses: List of predicted gloss tokens (for BLEU)
        
        Returns:
            Dict containing all metrics
        """
        results = {}
        
        # Pose metrics
        dtw_scores = []
        mse_scores = []
        accel_errors = []
        
        for gt, pred in zip(gt_poses, pred_poses):
            dtw_scores.append(self.compute_dtw(gt, pred))
            mse_scores.append(self.compute_pose_mse(gt, pred))
            accel_errors.append(self.compute_acceleration_error(gt, pred))
        
        results['dtw'] = np.mean(dtw_scores)
        results['pose_mse'] = np.mean(mse_scores)
        results['acceleration_error'] = np.mean(accel_errors)
        
        # Text metrics (if provided)
        if gt_texts and pred_texts:
            rouge = self.compute_rouge(gt_texts, pred_texts)
            results.update(rouge)
        
        # Gloss metrics (if provided)
        if gt_glosses and pred_glosses:
            bleu = self.compute_bleu(gt_glosses, pred_glosses)
            results.update(bleu)
        
        return results
    
    # ==================== UTILITY ====================
    
    def print_results(self, results: Dict[str, float], title: str = "Evaluation Results"):
        """Pretty print evaluation results"""
        print("=" * 70)
        print(f"{title:^70}")
        print("=" * 70)
        
        # Group by category
        text_metrics = {k: v for k, v in results.items() if 'bleu' in k or 'rouge' in k}
        pose_metrics = {k: v for k, v in results.items() if k not in text_metrics}
        
        if text_metrics:
            print("\n📝 Text/Gloss Metrics:")
            for k, v in sorted(text_metrics.items()):
                print(f"  {k:20s}: {v:.4f}")
        
        if pose_metrics:
            print("\n🤸 Pose Metrics:")
            for k, v in sorted(pose_metrics.items()):
                print(f"  {k:20s}: {v:.4f}")
        
        print("=" * 70)


# ==================== STANDALONE FUNCTIONS ====================

def compute_bleu_score(references: List[List[str]], hypotheses: List[List[str]]) -> float:
    """Quick BLEU-4 computation"""
    evaluator = SignLanguageEvaluator()
    return evaluator.compute_bleu(references, hypotheses)['bleu-4']


def compute_dtw_distance(gt_poses: np.ndarray, pred_poses: np.ndarray) -> float:
    """Quick DTW computation"""
    evaluator = SignLanguageEvaluator()
    return evaluator.compute_dtw(gt_poses, pred_poses)


if __name__ == "__main__":
    # Example usage
    evaluator = SignLanguageEvaluator()
    
    # Test text metrics
    refs = [["hello world"], ["how are you"]]
    hyps = [["hello there"], ["how are you"]]
    
    bleu = evaluator.compute_bleu(refs, hyps)
    print("BLEU scores:", bleu)
    
    # Test pose metrics
    gt_pose = np.random.randn(100, 75, 2)
    pred_pose = gt_pose + np.random.randn(100, 75, 2) * 0.1
    
    dtw = evaluator.compute_dtw(gt_pose, pred_pose)
    print(f"DTW distance: {dtw:.4f}")
