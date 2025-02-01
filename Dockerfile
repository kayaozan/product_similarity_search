# A simple Dockerfile to run the scripts in a container.

FROM pytorch/pytorch
# Necessary packages for image manipulations.
RUN apt-get update -y && apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6
# Install required Python Packages.
RUN pip install ipython ipykernel pyodbc pandas numpy ultralytics streamlit streamlit-extras streamlit-image-select
# Expose port to run streamlit.
EXPOSE 8501
