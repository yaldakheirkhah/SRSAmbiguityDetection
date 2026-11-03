from pathlib import Path
import sys
import pytest
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from ambifix_hitl.storage import Store
from ambifix_hitl.predictors import MockPredictor
from ambifix_hitl.service import ReviewService


@pytest.fixture
def store(tmp_path):return Store(tmp_path/"reviews.sqlite3")


@pytest.fixture
def service(store):return ReviewService(MockPredictor(),store)


@pytest.fixture
def prediction(service):return service.analyze("  The violet rehearsal lamp shall flash after Preview.\n  ")[0]
