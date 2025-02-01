FROM pytorch/pytorch
RUN apt-get update -y && apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6
RUN pip install ipython ipykernel pyodbc pandas numpy ultralytics streamlit streamlit-extras streamlit-image-select
EXPOSE 8501