import json

class MetadataRepository:

    def retrieve(self,model_id):

        with open("models/sample_model.json","r") as file:
            metadata=json.load(file)

        print("\n========== STEP 2 ==========")
        print("Retrieve Model Metadata\n")

        if metadata["model_id"]==model_id:

            print("Metadata Found\n")

            for key,value in metadata.items():

                if isinstance(value,list):
                    print(key," : ",", ".join(value))
                else:
                    print(key," : ",value)

            return metadata

        print("Metadata Not Found")
        return None