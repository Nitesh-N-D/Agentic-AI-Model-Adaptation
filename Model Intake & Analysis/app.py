from intake import ModelReceiver
from metadata import MetadataRepository
from validator import MetadataValidator
from catalog import ModelCatalog


receiver=ModelReceiver()

analysis=receiver.receive()

repository=MetadataRepository()

metadata=repository.retrieve(analysis["model_id"])

if metadata:

    validator=MetadataValidator()

    if validator.validate(metadata):

        catalog=ModelCatalog()

        catalog.forward(metadata)