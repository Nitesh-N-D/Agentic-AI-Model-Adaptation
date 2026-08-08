class MetadataValidator:

    REQUIRED_FIELDS=[
        "model_id",
        "model_name",
        "version",
        "framework",
        "owner",
        "purpose",
        "dependencies"
    ]

    def validate(self,metadata):

        print("\n========== STEP 3 ==========")
        print("Validate Metadata\n")

        for field in self.REQUIRED_FIELDS:

            if field not in metadata:
                print(field,"Missing")
                return False

            if metadata[field]=="":
                print(field,"Empty")
                return False

        print("Metadata Validation Successful")

        return True