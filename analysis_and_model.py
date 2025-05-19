import streamlit as st
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from ucimlrepo import fetch_ucirepo


def load_data():
    try:
        # Загрузка датасета
        dataset = fetch_ucirepo(id=601)
        data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        return data
    except:
        st.error("Ошибка загрузки данных. Пожалуйста, проверьте подключение к интернету.")
        return None


def preprocess_data(data):
    columns_to_drop = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    for col in columns_to_drop:
        if col in data.columns:
            data = data.drop(columns=[col])
    # Переименовываем столбцы для единообразия
    column_mapping = {
            'Air temperature [K]': 'Air temperature',
            'Process temperature [K]': 'Process temperature',
            'Rotational speed [rpm]': 'Rotational speed',
            'Torque [Nm]': 'Torque',
            'Tool wear [min]': 'Tool wear'
        }
    data = data.rename(columns=column_mapping)

    # Преобразование категориальной переменной Type в числовую
    data['Type'] = LabelEncoder().fit_transform(data['Type'])

    return data


def train_model(X_train, y_train, model_type):
    if model_type == "Логистическая регрессия":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "Случайный лес":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "XGBoost":
        model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    elif model_type == "Support Vector Machine (SVM)":
        model = SVC(kernel='linear', probability=True, random_state=42)

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    return accuracy, conf_matrix, class_report, roc_auc


def plot_roc_curve(y_test, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label='ROC кривая')
    ax.plot([0, 1], [0, 1], linestyle='--', label='Случайный классификатор')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC кривая')
    ax.legend()
    return fig


def analysis_and_model_page():
    st.title("Анализ данных и модель предиктивного обслуживания")

    # Загрузка данных
    st.header("1. Загрузка данных")
    data = load_data()

    if data is not None:
        st.success("Данные успешно загружены!")
        st.write("Первые 5 строк датасета:", data.head())

        # Предобработка данных
        st.header("2. Предобработка данных")
        data = preprocess_data(data)
        st.write("Данные после предобработки:", data.head())

        # Разделение данных
        st.header("3. Разделение данных")
        X = data.drop(columns=['Machine failure'])
        y = data['Machine failure']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.write(f"Обучающая выборка: {X_train.shape[0]} записей")
        st.write(f"Тестовая выборка: {X_test.shape[0]} записей")

        # Масштабирование данных
        scaler = StandardScaler()
        numerical_features = ['Air temperature', 'Process temperature',
                              'Rotational speed', 'Torque', 'Tool wear']
        X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
        X_test[numerical_features] = scaler.transform(X_test[numerical_features])

        # Выбор модели
        st.header("4. Обучение модели")
        model_type = st.selectbox("Выберите модель",
                                  ["Логистическая регрессия", "Случайный лес", "XGBoost",
                                   "Support Vector Machine (SVM)"])

        if st.button("Обучить модель"):
            model = train_model(X_train, y_train, model_type)
            accuracy, conf_matrix, class_report, roc_auc = evaluate_model(model, X_test, y_test)

            # Отображение результатов
            st.header("5. Оценка модели")
            st.subheader("Метрики качества")
            st.write(f"Accuracy: {accuracy:.4f}")
            st.write(f"ROC-AUC: {roc_auc:.4f}")

            st.subheader("Матрица ошибок")
            fig, ax = plt.subplots()
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

            st.subheader("Отчет классификации")
            st.text(class_report)

            st.subheader("ROC кривая")
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            roc_fig = plot_roc_curve(y_test, y_pred_proba)
            st.pyplot(roc_fig)

            # Сохранение модели для предсказаний
            st.session_state.model = model
            st.session_state.scaler = scaler

        # Предсказание на новых данных
        st.header("6. Предсказание на новых данных")
        st.write("Введите параметры оборудования для предсказания:")

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                type_input = st.selectbox("Тип оборудования", ["L", "M", "H"])
                air_temp = st.number_input("Температура воздуха [K]", value=300.0)
                process_temp = st.number_input("Температура процесса [K]", value=310.0)

            with col2:
                rotational_speed = st.number_input("Скорость вращения [rpm]", value=1500)
                torque = st.number_input("Крутящий момент [Nm]", value=40.0)
                tool_wear = st.number_input("Износ инструмента [min]", value=0)

            submit_button = st.form_submit_button("Сделать предсказание")

            if submit_button and 'model' in st.session_state:
                # Преобразование введенных данных
                type_mapping = {"L": 0, "M": 1, "H": 2}
                input_data = pd.DataFrame({
                    'Type': [type_mapping[type_input]],
                    'Air temperature': [air_temp],
                    'Process temperature': [process_temp],
                    'Rotational speed': [rotational_speed],
                    'Torque': [torque],
                    'Tool wear': [tool_wear]
                })

                # Масштабирование
                input_data[numerical_features] = st.session_state.scaler.transform(input_data[numerical_features])

                # Предсказание
                prediction = st.session_state.model.predict(input_data)
                prediction_proba = st.session_state.model.predict_proba(input_data)[:, 1]

                # Отображение результатов
                st.subheader("Результат предсказания")
                if prediction[0] == 1:
                    st.error(f"Прогнозируется отказ оборудования (вероятность: {prediction_proba[0]:.2%})")
                else:
                    st.success(f"Отказ оборудования не прогнозируется (вероятность: {1 - prediction_proba[0]:.2%})")


analysis_and_model_page()