import copy
from backend.scoring.rules import BASE_ESCORE
from backend.rnn.score import evaluate_routine

class AppLogic:
    def __init__(self):
        self.e_score = BASE_ESCORE
        self.errors_by_frame = {}
        self.error_frames_list = []
        self.undo_stack = []
        self.redo_stack = []

    def load_json_data(self, filename):
        results = evaluate_routine(filename, window=40, step=20)
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.e_score = BASE_ESCORE
        self.errors_by_frame.clear()
        self.error_frames_list.clear()
        
        for res in results:
            frame = res["global_peak"]
            self.error_frames_list.append(frame)
            
            reasons_list = []
            for b_str in res["breakdown"]:
                try:
                    val_str = b_str.split("Applying -")[1].split(" deduction")[0]
                    penalty_val = float(val_str)
                except:
                    penalty_val = 0.0
                
                reasons_list.append({"text": b_str, "penalty": penalty_val, "status": "pending"})
                
            self.errors_by_frame[frame] = {
                "acrobatic": res["acrobatic"],
                "confidence": res.get("confidence", 0.0), 
                "position": res["position"],
                "breakdown": res["breakdown"], 
                "reasons": reasons_list
            }
            
        self.error_frames_list.sort()
        return len(self.error_frames_list)

    def save_state(self):
        self.undo_stack.append({
            'e_score': self.e_score,
            'errors_by_frame': copy.deepcopy(self.errors_by_frame)
        })
        self.redo_stack.clear()

    def undo(self):
        if not self.undo_stack: return False
        self.redo_stack.append({'e_score': self.e_score, 'errors_by_frame': copy.deepcopy(self.errors_by_frame)})
        state = self.undo_stack.pop()
        self.e_score = state['e_score']
        self.errors_by_frame = state['errors_by_frame']
        return True

    def redo(self):
        if not self.redo_stack: return False
        self.undo_stack.append({'e_score': self.e_score, 'errors_by_frame': copy.deepcopy(self.errors_by_frame)})
        state = self.redo_stack.pop()
        self.e_score = state['e_score']
        self.errors_by_frame = state['errors_by_frame']
        return True

    def accept_deduction(self, frame_idx, reason_idx):
        self.save_state()
        reason = self.errors_by_frame[frame_idx]["reasons"][reason_idx]
        reason["status"] = "accepted"
        self.e_score -= reason["penalty"]

    def reject_deduction(self, frame_idx, reason_idx):
        self.save_state()
        self.errors_by_frame[frame_idx]["reasons"][reason_idx]["status"] = "rejected"

    def reject_all_in_frame(self, frame_idx):
        if frame_idx not in self.errors_by_frame: return False
        self.save_state()
        data = self.errors_by_frame[frame_idx]
        
        for reason in data["reasons"]:
            if reason["status"] == "accepted":
                self.e_score += reason["penalty"]
            reason["status"] = "rejected"
            
        data["acrobatic"] = "Transition"
        return True