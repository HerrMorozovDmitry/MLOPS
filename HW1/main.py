from flask import Flask
from flask_restx import Api, Resource, fields
from models import Model

"""
В константах указан путь к датасету и необходимые столбцы для дальнейшего обучения.
Текущая реализация подразумевает, что выдается предсказание для целевой переменной из train-файла.
"""


MODELS = dict()
DATASET_PATH = "titanic/train.csv"
TARGET = "Survived"
IGNORE_FEATURES = ["PassengerId", "Name", "Ticket", "Cabin"]

app = Flask(__name__)
api = Api(app)


@api.route("/models/types")
class ModelTypes(Resource):
    @api.doc(responses={201: "Success"})
    def get(self):
        model_types = {
            "LogReg": "LogisticRegression",
            "SVM": "SVC",
            "DecisionTree": "DecisionTreeClassifier",
            "LGBT": "LGBMClassifier",
        }
        return model_types, 201


@api.route("/models/list")
class ModelList(Resource):
    @api.doc(responses={201: "Success"})
    def get(self):
        model_dict = {
            name: {
                "type": MODELS[name].type,
                "trained": MODELS[name].trained,
                "accuracy": MODELS[name].accuracy,
            }
            for name in MODELS.keys()
        }
        return model_dict, 201


ml_model = api.model(
    "model",
    {
        "name": fields.String(
            required=True,
            title="Model Name",
            description="Unique Model Identifier;",
            default="1",
        ),
        "type": fields.String(
            required=True,
            title="Model Type",
            description="'LogReg', 'SVM', 'DecisionTree' or 'LGBT';",
            default="LogReg",
        ),
        "params": fields.String(
            required=True,
            title="Model Params",
            description="Parameters Dictionary for Model Training;",
            default="{'random_state': 54}",
        ),
    },
)


@api.route("/models/add")
class ModelAdd(Resource):
    @api.expect(ml_model)
    @api.doc(
        responses={
            201: "Success",
            401: "Params Error",
            402: "Model Initialization Error",
            403: "Model Already Exists",
        }
    )
    def post(self):
        name = api.payload["name"]
        type = api.payload["type"]
        params = api.payload["params"]
        try:
            params = eval(params)
        except:
            return {"status": "Failed", "message": "Params Error"}, 401
        if name in MODELS.keys():
            return {"status": "Failed", "message": "Model Already Exists"}, 403
        else:
            try:
                MODELS[name] = Model(
                    type=type,
                    params=params,
                    dataset_path=DATASET_PATH,
                    target=TARGET,
                    drop_cols=IGNORE_FEATURES,
                )
                return {"status": "Success", "message": "Model Created"}, 201
            except Exception:
                return {
                    "status": "Failed",
                    "message": "Model Initialization Error",
                }, 402


parser = api.parser()
parser.add_argument("name", type=str, required=True, help="Model Name", location="args")


@api.route("/models/remove")
class ModelDelete(Resource):
    @api.expect(parser)
    @api.doc(responses={201: "Success", 404: "Model Does Not Exist"})
    def delete(self):
        name = parser.parse_args()["name"]
        if name in MODELS.keys():
            MODELS.pop(name)
            return {"status": "Success", "message": "Model was Deleted"}, 201
        else:
            return {"status": "Failed", "message": "Model Does Not Exist"}, 404


@api.route("/models/fit")
class ModelFit(Resource):
    @api.expect(parser)
    @api.doc(
        responses={
            201: "Success",
            404: "Model Does Not Exist",
            405: "Model Training Error",
        }
    )
    def post(self):
        name = parser.parse_args()["name"]
        if name in MODELS.keys():
            try:
                return {"status": "Success", "message": MODELS[name].fit()}, 201
            except Exception:
                return {
                    "status": "Failed",
                    "message": "Model Training Error",
                }, 405
        else:
            return {
                "status": "Failed",
                "message": "Model Does Not Exist",
            }, 404


@api.route("/models/prediction")
class ModelPrediction(Resource):
    @api.expect(parser)
    @api.doc(
        responses={
            201: "Success",
            404: "Model Does Not Exist",
            406: "Model is not Trained",
        }
    )
    def get(self):
        name = parser.parse_args()["name"]
        if name in MODELS.keys():
            try:
                return {"status": "Success", "message": MODELS[name].predict()}, 201
            except Exception:
                return {
                    "status": "Failed",
                    "message": "Model is not Trained",
                }, 406
        else:
            return {
                "status": "Failed",
                "message": "Model Does Not Exist",
            }, 404


if __name__ == "__main__":
    app.run()
