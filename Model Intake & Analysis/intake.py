import json

class ModelReceiver:

    def receive(self):

        with open("analysis_result.json","r") as file:
            analysis=json.load(file)

        print("\n========== STEP 1 ==========")
        print("Receive Model ID + Analysis Result\n")

        print("Model ID        :",analysis["model_id"])
        print("Drift Status    :",analysis["drift_status"])
        print("Conflict Status :",analysis["conflict_status"])
        print("Cascade Status  :",analysis["cascade_status"])

        return analysis