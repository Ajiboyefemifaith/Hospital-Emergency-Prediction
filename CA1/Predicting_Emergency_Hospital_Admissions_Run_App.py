{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "38b7ace0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2"
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import subprocess\n",
    "\n",
    "# Step 1: Run data download and preprocessing\n",
    "subprocess.call(['python', 'Predicting Emergency Hospital Admissions- Data download and Cleaning.ipynb'])\n",
    "\n",
    "# Step 2: Run data visualization script\n",
    "subprocess.call(['python', 'Predicting Emergency Hospital Admissions-Data Visualization and EDA.ipynb'])\n",
    "\n",
    "# Step 3: Run model building script\n",
    "subprocess.call(['python', 'Predicting Emergency Hospital Admissions- Model Building.ipynb'])\n",
    "\n",
    "# Step 4: Run the Streamlit app\n",
    "subprocess.call(['streamlit', 'run', 'Predicting Emergency Hospital Admissions- App.ipynb'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ed17951a",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
