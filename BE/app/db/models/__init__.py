# app/db/models/__init__.py

# Import base first if other models might depend on it implicitly
# (though usually not necessary for this specific problem)
# from ..base import Base # Assuming Base is in app/db/base.py

# Import your model classes
from .connections import Connection # Assuming you have this model based on datasets.py
from .cleaning_jobs import CleaningJob
from .datasets import Dataset
from .automl_sessions import AutoMLSession
from .finalized_models import FinalizedModel # Assuming you have this model
from .chatbot_sessions import ChatSessionState 
 # Assuming you have this model

# Optional: You can define __all__ if you want to control what 'from app.db.models import *' imports
# __all__ = [
#     "Connection",
#     "Dataset",
#     "AutoMLSession",
#     "AutoMLJob",
#     "FinalizedModel",
#     "CleaningJob",
# ]