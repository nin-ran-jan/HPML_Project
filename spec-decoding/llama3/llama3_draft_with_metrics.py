import time, torch
from transformers.generation.candidate_generator import AssistedCandidateGenerator

# add timing, acceptance stats and rollbacks
class InstrumentedDraft(AssistedCandidateGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accepted = self.rejected = self.rollbacks = 0
        self.latencies = []         

    def get_candidates(self, input_ids):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()

        ids, logits = super().get_candidates(input_ids)

        end.record(); torch.cuda.synchronize()
        self.latencies.append(start.elapsed_time(end)) 
        return ids, logits

    def update_candidate_strategy(self, input_ids, scores, num_matches):
        super().update_candidate_strategy(input_ids, scores, num_matches)
        self.accepted += num_matches
        self.rejected += scores.shape[1] - 1 - num_matches
        if num_matches < scores.shape[1] - 1:
            self.rollbacks += 1
