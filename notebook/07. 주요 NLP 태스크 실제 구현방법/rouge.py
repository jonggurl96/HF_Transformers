r"""
영어와 비교했을 때 한국어 문장성분의 특성 상 순서보다 조사가 중요함
따라서 순서가 아닌 조합으로 성능을 판단할 수 있는 ROUGE-U, ROUGE-SU가 적절해보임
	+ RDASS
"""

from collections import defaultdict
from itertools import combinations
from tqdm.auto import tqdm
from typing import Dict, Tuple

import numpy as np
import statistics as st

class RougeScore(dict):
	def __init__(self, scores: np.ndarray = None, obj: dict = None):
		super().__init__()
		
		if obj is None and scores is None:
			raise Exception("Must be None only one of scores or obj.")
		
		if obj and scores:
			raise Exception("Must be None either scores or obj.")
		
		if scores is not None:
			self["min"] = np.min(scores)
			self["max"] = np.max(scores)
			self["avg"] = np.mean(scores, dtype = float)
		
		if obj is not None:
			self["min"] = obj["min"] if obj["min"] is not None else 0.0
			self["max"] = obj["max"] if obj["max"] is not None else 1.0
			self["avg"] = obj["avg"] if obj["avg"] is not None else self.max
			
	
	@property
	def min(self) -> float:
		return self["min"]
	
	
	@property
	def avg(self) -> float:
		return self["avg"]
	
	
	@property
	def max(self) -> float:
		return self["max"]
	

rouge_names = ["rouge-precision", "rouge-recall", "rouge-f1", "rouge-s", "rouge-su"]


def replace_sentence(sentence: str) -> str:
	return sentence.replace(".", "")


def split_to_tokens(hypothesis: str, reference: str) -> Tuple[np.ndarray, np.ndarray]:
	hypothesis = replace_sentence(hypothesis)
	reference = replace_sentence(reference)
	return np.array(hypothesis.split(), dtype = str), np.array(reference.split(), dtype = str)


def get_matched_cnt(basis: np.ndarray, compared: np.ndarray) -> int:
	cnt = 0
	for compared_token in compared:
		if compared_token in basis:
			cnt += 1
	return cnt


def recall(hypothesis: str, reference: str) -> float:
	token_hypo, token_ref = split_to_tokens(hypothesis, reference)
	return get_matched_cnt(token_hypo, token_ref) / len(token_ref)


def precision(hypothesis: str, reference: str) -> float:
	token_hypo, token_ref = split_to_tokens(hypothesis, reference)
	return get_matched_cnt(token_ref, token_hypo) / len(token_hypo)



def rouge_precision(hypotheses: np.ndarray, references: np.ndarray, show_tqdm: bool) -> RougeScore:
	r"""
	
	:param hypotheses: decoded model outputs
	:param references: labels
	:param show_tqdm: progress bar showing
	:return: 모델 예측에 포함된 단어가 label에 포함된 확률의 산술평균
	"""
	precision_array = np.array([], dtype = float)
	
	pb = None
	if show_tqdm:
		print(">>> ROUGE-PRECISION")
		pb = tqdm(range(references.size))
	
	for hypothesis, reference_list in zip(hypotheses, references):
		for reference in reference_list:
			precision_array = np.append(precision_array, precision(hypothesis, reference))
			if pb is not None:
				pb.update(1)
	
	return RougeScore(precision_array)


def rouge_recall(hypotheses: np.ndarray, references: np.ndarray, show_tqdm: bool) -> RougeScore:
	r"""
	
	:param hypotheses: decoded model outputs
	:param references: labels
	:param show_tqdm: progress bar showing
	:return: label 단어가 모델 예측에 포함된 확률의 산술평균
	"""
	recall_array = np.array([], dtype = float)
	
	pb = None
	if show_tqdm:
		print(">>> ROUGE-RECALL")
		pb = tqdm(range(references.size))
	
	for hypothesis, reference_list in zip(hypotheses, references):
		for reference in reference_list:
			recall_array = np.append(recall_array, recall(hypothesis, reference))
			if pb is not None:
				pb.update(1)
	
	return RougeScore(recall_array)


def rouge_f1(hypotheses: np.ndarray = None, references: np.ndarray = None, precision = None, recall = None, show_tqdm: bool = False) -> RougeScore:
	r"""
	precision과 recall의 조화평균
	:param hypotheses: decoded model outputs
	:param references: labels
	:param precision: default None
	:param recall: default None
	:param show_tqdm: progress bar showing default False
	:return: rouge-1-f1
	"""
	
	new_precision = precision if precision is not None else rouge_precision(hypotheses, references, show_tqdm)
	new_recall = recall if recall is not None else rouge_recall(hypotheses, references, show_tqdm)
	
	min_precision = new_precision.min
	min_recall = new_recall.min
	
	return RougeScore(obj = {
			"min": st.harmonic_mean([min_precision, min_recall]) if min_precision + min_recall > 0 else 0.0,
			"avg": st.harmonic_mean([new_precision.avg, new_recall.avg]),
			"max": st.harmonic_mean([new_precision.max, new_recall.max])
	})



def get_matched_combs(sentence: str, skip_gram: int = 2) -> np.ndarray:
	r"""
	ROUGE-S와 ROUGE-SU에서 사용할 skip_gram 크기의 토큰 쌍들 목록
	:param sentence: combination 생성할 문장
	:param skip_gram: window size
	:return: sentence로 생성한 skip_gram 크기의 combinations
	"""
	sentence = replace_sentence(sentence)
	matched_comb_list = np.array([], dtype = str)
	matched_comb_list = np.append(matched_comb_list, list(combinations(sentence.split(), skip_gram)))
	matched_comb_list = np.array(np.array_split(matched_comb_list, len(matched_comb_list) // skip_gram))
	return matched_comb_list


def s_recall_cnt(basis: np.ndarray, compared: np.ndarray) -> int:
	cnt = 0
	
	for base in basis:
		for comp in compared:
			for b, c in zip(base, comp):
				if b != c:
					cnt -= 1
					break
			cnt += 1
	
	return cnt


def rouge_s(hypotheses: np.ndarray, references: np.ndarray, show_tqdm: bool, skip_gram: int = 2) -> RougeScore:
	r"""
	skip_gram 크기의 토큰을 한 쌍으로 묶어서 ROUGE Score를 계산
	:param hypotheses: decoded model outputs
	:param references: labels
	:param show_tqdm: progress bar showing
	:param skip_gram: window size default 2
	:return: rouge-s 산술평균
	"""
	rouge_s_array = np.array([], dtype = float)
	
	pb = None
	if show_tqdm:
		print(">>> ROUGE-S")
		pb = tqdm(range(references.size))
	
	for hypothesis, reference_list in zip(hypotheses, references):
		matched_hypos = get_matched_combs(hypothesis)
		for reference in reference_list:
			matched_refs = get_matched_combs(reference, skip_gram)
			rouge_s_array = np.append(rouge_s_array, s_recall_cnt(matched_hypos, matched_refs) / len(matched_refs))
			if pb is not None:
				pb.update(1)
	
	return RougeScore(rouge_s_array)


def list2set(word_list: np.ndarray):
	r"""
	
	:param word_list: ROUGE-S에서 사용할 토큰 쌍 목록
	:return: 목록의 모든 원소를 set으로 변환 후 리스트로 반환
	"""
	
	if word_list.ndim < 2:
		raise Exception("At least dimension-2 required.")
	
	ret_val = set()
	
	for words in word_list:
		for word in words:
			ret_val.add(word)
	
	return np.array(list(ret_val), dtype = str)


def rouge_su(hypotheses: np.ndarray, references: np.ndarray, show_tqdm: bool, skip_gram: int = 2) -> RougeScore:
	r"""
	ROUGE-S + Unigram
	:param hypotheses: decoded model outputs
	:param references: labels
	:param show_tqdm: progress bar showing
	:param skip_gram: window size default 2
	:return: rouge-su 산술평균
	"""
	rouge_su_array = np.array([], dtype = float)
	
	pb = None
	if show_tqdm:
		print(">>> ROUGE-SU")
		pb = tqdm(range(references.size))
	
	for hypothesis, reference_list in zip(hypotheses, references):
		matched_hypos = get_matched_combs(hypothesis, skip_gram)
		matched_hypos_unit = list2set(matched_hypos)
		
		for reference in reference_list:
			matched_refs = get_matched_combs(reference, skip_gram)
			matched_refs_unit = list2set(matched_refs)
			
			matched_recall_cnt = s_recall_cnt(matched_hypos, matched_refs) + get_matched_cnt(matched_hypos_unit, matched_refs_unit)
			rouge_su_array = np.append(rouge_su_array, matched_recall_cnt / (len(matched_refs) + len(matched_refs_unit)))
			if pb is not None:
				pb.update(1)
	
	return RougeScore(rouge_su_array)


def compute_metric(
		hypotheses: np.ndarray,
		references: np.ndarray,
		skip_gram: int = 2,
		use_rouge_s: bool = True,
		no_precision_recall: bool = True,
		show_tqdm: bool = False
) -> Dict:
	r"""
	ROUGE 한국어 못찾겠어서 직접 만듦
	:param hypotheses: 모델 추론 결과 가설
	:param references: 정답 문장
	:param skip_gram: window size, default 2
	:param use_rouge_s: ROUGE-S, ROUGE-SU 사용 여부, default True
	:param no_precision_recall: precision, recall 반환 여부, default True
	:param show_tqdm: progress bar showing, default False
	:return:
	"""
	if hypotheses.shape[0] != references.shape[0]:
		raise Exception("inferenced text length is different with labels")
	
	if skip_gram < 2:
		use_rouge_s = False
	
	new_references = np.expand_dims(references, axis = -1) if type(references[0]) != type(np.array([])) else references
	result = defaultdict(RougeScore)
	
	if no_precision_recall:
		result["rouge-f1"] = rouge_f1(hypotheses = hypotheses, references = new_references, show_tqdm = show_tqdm)
	else:
		result["rouge-precision"] = rouge_precision(hypotheses, new_references, show_tqdm)
		result["rouge-recall"] = rouge_recall(hypotheses, new_references, show_tqdm)
		result["rouge-f1"] = rouge_f1(precision = result["rouge-precision"], recall = result["rouge-recall"])
	
	if use_rouge_s:
		result["rouge-s"] = rouge_s(hypotheses, new_references, show_tqdm, skip_gram)
		result["rouge-su"] = rouge_su(hypotheses, new_references, show_tqdm, skip_gram)
	
	return result


if __name__ == "__main__":
	generated_summary = np.array([
			"던졌다 공을 류현진이",
			"나는 밥을 먹을 예정이다.",
			"나는 새 헤드셋을 구매했다.",
			"널 지킬 사람을 몰라",
			"말을 타는 서부의 총잡이",
			"모래바람 날리는 소떼를 몰거야",
			"나와 닮았을 너의 나락으로"
	])
	reference_summary = np.array([
			["류현진이 공을 던졌다", "류현진이 삽을 던졌다"],
			["나는 밥을 지을 예정이다.", "나는 밥을 먹기로 했다."],
			["나는 어제 새 헤드셋을 구매했다.", "나는 새 헤드셋을 구매할 예정이다."],
			["널 지킬 쌈자를 몰라", "지킬과 하이드를 몰라"],
			["칼을 가는 서부의 칼잡이", "총을 닦는 북부의 총잡이"],
			["칼바람 협곡의 미친 소떼", "휘파람 불며 양들을 몰거야"],
			["너와 닮았을 나의 나락으로", "나와 닮았을 너의 나락으로"]
	])
	
	rouge_scores = compute_metric(generated_summary, reference_summary, no_precision_recall = False)
	for k, v in rouge_scores.items():
		print(f"\n>>>key: {k}")
		print(f"value: {v}")
	