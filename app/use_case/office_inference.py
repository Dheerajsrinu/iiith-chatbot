import uuid
import pandas as pd
from joblib import load
from app import model_store
class OfficeInference():

    def run_inference(
            self,
            score: int
        ):
        questions_array =[
            {
                "mod_catboost":[{
                    "Q22_7": "High Impact",
                    "Q20_6": "Strongly disagree"
                }]
            }
        ]

        if score >= 18 and score <=22:
            input_dict = questions_array[0]["mod_catboost"]
        df_new = pd.DataFrame(input_dict)

        metadata = load("models/office/mod_catboost_metadata.joblib")
        
        print(df_new)
        y_pred = model_store.mod_catboost.predict(df_new)
        print(y_pred)
        y_proba = model_store.mod_catboost.predict_proba(df_new)[:, 1]
        print(y_proba)
        subgroup = ['Navy' if p > 0.476 else 'Royal' for p in y_proba]
        print(subgroup)

        return subgroup

    def execute(self, event):
        score = event["score"]
        execute_response = self.run_inference(score=score)
        return execute_response
