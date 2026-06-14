# Use a minimal base image with Python 3.9 installed
FROM python:3.9

# Create a non-root user and set its home directory
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
# Ensure the user's local bin is in PATH for pip installs
ENV PATH=$HOME/.local/bin:$PATH

# Set the working directory for the application files within the user's home
# This will be /home/user/app on the Hugging Face Space
WORKDIR $HOME/app

# Copy the requirements.txt first to leverage Docker's build cache
# When the 'deployment' folder from the GitHub repo is the build context,
# requirements.txt is directly at its root.
COPY requirements.txt ./ 

# Install Python dependencies as the non-root user
# --no-cache-dir reduces image size
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application files from the current build context root
# (which is 'tourism_project_updated/deployment' from your git repo)
COPY app.py ./ 

# Define the command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
