# Домашнее задание 1
* Задание выполнялось на основе классического датасета Titanic (https://www.kaggle.com/competitions/titanic/data?select=train.csv).
* Был реализован вариант на REST API с использованием Swagger. Соответственно, все тестировал через браузер по http://127.0.0.1:5000/.
* В **environment.yml** указано используемое окружение. Чтобы его развернуть, можно воспользоваться командой `conda env create -f environment.yml`.
* Программа запускается просто через `python main.py`.
* Из реализованных методов:
  * get /models/types — выводит все поддерживаемые типы моделей.
  * get /models/list — выводит все определенные модели.
  * post /models/add — определяет новую модель с заданными параметрами.
  * delete /models/remove — удалаяет существующую модель.
  * post /models/fit — обучает определенную модель и рассчитывает ее точность.
  * get /models/prediction — возвращает предсказание от обученной модели.
* В качестве примеров можно позапускать следующие варианты моделей:
1. {
  "name": "1",
  "type": "LogReg",
  "params": "{'random_state': 54, 'max_iter': 100}"
}
2. {
  "name": "2",
  "type": "SVM",
  "params": "{'random_state': 54, 'kernel': 'linear'}"
}
3. {
  "name": "3",
  "type": "DecisionTree",
  "params": "{'random_state': 54, 'max_depth': 50}"
}
4. {
  "name": "4",
  "type": "LGBT",
  "params": "{'random_state': 54, 'num_iterations': 100, 'learning_rate': 0.2}"
}
