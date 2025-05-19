import streamlit as st


def main():
    st.sidebar.title("Навигация")
    page = st.sidebar.radio(
        "Выберите страницу",
        ["Анализ и модель", "Презентация"]
    )

    if page == "Анализ и модель":
        from analysis_and_model import analysis_and_model_page
        analysis_and_model_page()
    else:
        from presentation import presentation_page
        presentation_page()


if __name__ == "__main__":
    main()