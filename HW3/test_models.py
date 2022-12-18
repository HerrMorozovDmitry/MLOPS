import ast
import json

import pytest

from models import Model

MODEL_TYPE = "LogReg"
PARAMS = {"random_state": 54}
DATASET_PATH = "titanic/train.csv"
TARGET = "Survived"
IGNORE_FEATURES = ["PassengerId", "Name", "Ticket", "Cabin"]

# Задаем возможные конфигурации моделей
@pytest.fixture(
    scope="function",
    params=[
        ("LogReg", {"random_state": 1337}, "titanic/train.csv", "Survived", ["PassengerId", "Name", "Ticket", "Cabin"]),
        ("SVM", {"random_state": 42}, "titanic/train.csv", "Survived", ["PassengerId", "Name", "Ticket", "Cabin", "Fare"]),
        ("DecisionTree", {"random_state": 54}, "titanic/train.csv", "Survived", ["Age", "Name", "Ticket", "Cabin"]),
    ]
)
def init_model(request):
    return Model(*request.param)

# Проверяем, что при абракадабре возвращается исключение 
def test_init_raises():
    with pytest.raises(Exception):
        Model("LolKek", PARAMS, DATASET_PATH, TARGET, IGNORE_FEATURES)

# Проверяем, что подается модель из допустимого перечня
def test_type(init_model):
    assert init_model.type in ["LogReg", "SVM", "DecisionTree"]

# Проверяем, что отрабатывает обучение
def test_fit_trained(init_model):
    model = init_model
    model.fit()
    assert model.trained == True

# Проверяем, что модель возвращает адекватную точность
def test_fit_accuracy(init_model):
    model = init_model
    model.fit()
    assert 0 <= model.accuracy <= 1

# Проверяем, что модель выдает адекватный прогноз
def test_predict(init_model):
    model = init_model
    model.fit()
    assert len(model.predict()) > 0
    prediction = ast.literal_eval(str(json.loads(model.predict())))
    assert len(prediction) > 0
    assert min(prediction) == 0
    assert max(prediction) == 1