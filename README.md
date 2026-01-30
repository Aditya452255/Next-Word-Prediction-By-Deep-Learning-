# 🧠 Next Word Prediction with LSTM

A machine learning application that predicts the next word in a sentence using Long Short-Term Memory (LSTM) neural networks. Built with Streamlit for an interactive web interface.

## Features

- **LSTM-based Prediction**: Uses a trained LSTM model for accurate next-word predictions
- **Interactive Web Interface**: Built with Streamlit for easy interaction
- **Real-time Predictions**: Get instant predictions as you type
- **Tokenizer Integration**: Uses pre-trained tokenizer for text processing

## Project Structure

```
Next_word_pred/
├── app.py                    # Main Streamlit application
├── lstm_model.h5            # Pre-trained LSTM model
├── tokenizer.pkl            # Serialized tokenizer
├── max_len.pkl              # Maximum sequence length
├── qoute_dataset.csv        # Training dataset (quotes)
├── RNN_by_DL.ipynb          # Jupyter notebook with model training
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Microsoft C++ Redistributable (for TensorFlow on Windows)

### Setup Instructions

1. **Clone or download the project**
   ```bash
   cd Next_word_pred
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Access the app**
   - Open your browser and navigate to `http://localhost:8501`

## How It Works

1. **Input**: User enters a partial sentence or text
2. **Processing**: The text is tokenized and padded to match the model's expected input
3. **Prediction**: The LSTM model predicts the probability distribution for the next word
4. **Output**: The word with the highest probability is displayed as the prediction

## Technologies Used

- **Streamlit**: Web framework for creating interactive data applications
- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computing library
- **Pickle**: Python object serialization

## Model Details

- **Architecture**: LSTM (Long Short-Term Memory) neural network
- **Training Data**: Quote dataset
- **Features**: Text tokenization and sequence padding

## Deployment on Streamlit Cloud

### Option 1: Deploy on Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app"
4. Select your repository and main file (`app.py`)
5. Click "Deploy"

**Note**: Ensure all required files (model, tokenizer, max_len) are in the repository.

### Option 2: Deploy on Other Platforms

#### Heroku
```bash
# Create Procfile
echo "web: streamlit run app.py --logger.level=error" > Procfile

# Deploy
git push heroku main
```

#### PythonAnywhere or Other Servers
Follow platform-specific deployment guides for Streamlit applications.

## Configuration

The application uses the following configuration (in `.streamlit/config.toml`):
- Page layout: Centered
- Theme: Light
- Maximum file size: 200MB

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'streamlit'"
```bash
pip install streamlit
```

### Error: "Could not find the DLL(s) 'msvcp140.dll'"
Download and install [Microsoft C++ Redistributable](https://support.microsoft.com/help/2977003)

### Error: "Model file not found"
Ensure `lstm_model.h5`, `tokenizer.pkl`, and `max_len.pkl` are in the same directory as `app.py`

## Future Enhancements

- [ ] Support for multiple languages
- [ ] Batch prediction capability
- [ ] Model retraining interface
- [ ] Prediction confidence scores
- [ ] Usage analytics and logging
- [ ] Custom model upload

## License

This project is open source and available under the MIT License.

## Contact & Support

For issues or questions, please open an issue in the repository or contact the project maintainer.

---

**Built with ❤️ using Streamlit and TensorFlow**
