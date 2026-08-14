import numpy as np
import pandas as pd
import yfinance as yf
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# 1. Data Loading and Feature Engineering
# ---------------------------

def compute_indicators(df):
    """Add technical indicators to the dataframe."""
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Exponential Moving Averages
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

    # Bollinger Bands
    df['BB_mid'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_mid'] + 2 * bb_std
    df['BB_lower'] = df['BB_mid'] - 2 * bb_std
    df['BB_percent'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()

    # VWAP (approximation for daily: typical price * volume / cumulative volume)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()

    # Price returns (for additional context)
    df['Return_1'] = df['Close'].pct_change()
    df['Return_5'] = df['Close'].pct_change(periods=5)

    # Drop NaN rows created by indicators
    df.dropna(inplace=True)
    return df

def create_sequences(data, feature_columns, target_column, window_size=30, future_steps=1):
    """
    Create sliding window sequences for supervised learning.
    data: DataFrame with features
    feature_columns: list of column names to use as input features
    target_column: column name to predict (e.g., 'Close')
    window_size: number of past timesteps to include
    future_steps: number of future candles to predict (default 1)
    """
    X, y = [], []
    for i in range(len(data) - window_size - future_steps + 1):
        X.append(data[feature_columns].iloc[i:i+window_size].values)
        # Predict the close price 'future_steps' ahead
        y.append(data[target_column].iloc[i+window_size+future_steps-1])
    return np.array(X), np.array(y)

# ---------------------------
# 2. Genetic Algorithm Components
# ---------------------------

class Individual:
    """Represents a neural network candidate."""
    def __init__(self, architecture=None):
        if architecture is None:
            # Random initialization
            self.n_layers = random.randint(1, 3)  # number of hidden layers
            self.units = [random.choice([16, 32, 64, 128]) for _ in range(self.n_layers)]
            self.activation = random.choice(['relu', 'tanh'])
            self.dropout = random.uniform(0.0, 0.5)
            self.learning_rate = random.choice([0.01, 0.001, 0.0005, 0.0001])
            self.l2_reg = random.choice([0.0, 0.001, 0.0001])
        else:
            self.n_layers = architecture['n_layers']
            self.units = architecture['units']
            self.activation = architecture['activation']
            self.dropout = architecture['dropout']
            self.learning_rate = architecture['learning_rate']
            self.l2_reg = architecture['l2_reg']
        self.fitness = None

    def build_model(self, input_shape):
        """Build a Keras sequential model based on the individual's parameters."""
        model = Sequential()
        # Input layer (Dense) - we will flatten the 2D input (window_size, features)
        model.add(tf.keras.layers.Flatten(input_shape=input_shape))
        # Hidden layers
        for i in range(self.n_layers):
            model.add(Dense(self.units[i],
                            activation=self.activation,
                            kernel_regularizer=l2(self.l2_reg) if self.l2_reg > 0 else None))
            model.add(Dropout(self.dropout))
        # Output layer (single neuron for regression)
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def evaluate(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, patience=5):
        """Train the model and compute fitness (negative validation RMSE)."""
        model = self.build_model(X_train.shape[1:])
        early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0)
        history = model.fit(X_train, y_train,
                            validation_data=(X_val, y_val),
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[early_stop],
                            verbose=0)
        # Predict on validation set
        pred = model.predict(X_val, verbose=0).flatten()
        rmse = np.sqrt(np.mean((pred - y_val) ** 2))
        # Higher fitness = better (negative RMSE)
        self.fitness = -rmse
        return self.fitness

def crossover(parent1, parent2):
    """Create a child by mixing attributes of two parents."""
    child_arch = {
        'n_layers': random.choice([parent1.n_layers, parent2.n_layers]),
        'units': [],
        'activation': random.choice([parent1.activation, parent2.activation]),
        'dropout': (parent1.dropout + parent2.dropout) / 2,
        'learning_rate': random.choice([parent1.learning_rate, parent2.learning_rate]),
        'l2_reg': random.choice([parent1.l2_reg, parent2.l2_reg])
    }
    # Determine units list: combine from parents or random
    max_layers = max(parent1.n_layers, parent2.n_layers)
    for i in range(child_arch['n_layers']):
        choices = []
        if i < parent1.n_layers:
            choices.append(parent1.units[i])
        if i < parent2.n_layers:
            choices.append(parent2.units[i])
        if choices:
            child_arch['units'].append(random.choice(choices))
        else:
            child_arch['units'].append(random.choice([16, 32, 64, 128]))
    return Individual(child_arch)

def mutate(individual, mutation_rate=0.2):
    """Apply random mutations to an individual."""
    # Mutate number of layers
    if random.random() < mutation_rate:
        individual.n_layers = min(4, max(1, individual.n_layers + random.choice([-1, 1])))
        # Adjust units list length
        if len(individual.units) < individual.n_layers:
            individual.units.append(random.choice([16, 32, 64, 128]))
        elif len(individual.units) > individual.n_layers:
            individual.units = individual.units[:individual.n_layers]
    # Mutate units in a random layer
    if random.random() < mutation_rate and len(individual.units) > 0:
        idx = random.randrange(len(individual.units))
        individual.units[idx] = random.choice([16, 32, 64, 128])
    # Mutate activation
    if random.random() < mutation_rate:
        individual.activation = random.choice(['relu', 'tanh'])
    # Mutate dropout
    if random.random() < mutation_rate:
        individual.dropout = min(0.7, max(0.0, individual.dropout + random.uniform(-0.1, 0.1)))
    # Mutate learning rate
    if random.random() < mutation_rate:
        individual.learning_rate = random.choice([0.01, 0.001, 0.0005, 0.0001])
    # Mutate L2 regularization
    if random.random() < mutation_rate:
        individual.l2_reg = random.choice([0.0, 0.001, 0.0001])
    return individual

def evolve(X_train, y_train, X_val, y_val, pop_size=10, generations=20, keep_top=3):
    """Run the genetic algorithm."""
    population = [Individual() for _ in range(pop_size)]
    best_fitness_history = []

    for gen in range(generations):
        print(f"\n--- Generation {gen+1}/{generations} ---")
        # Evaluate all individuals
        for i, ind in enumerate(population):
            print(f"Evaluating individual {i+1}/{pop_size}...", end=' ')
            fitness = ind.evaluate(X_train, y_train, X_val, y_val, epochs=30, patience=3)
            print(f"Fitness (neg RMSE): {fitness:.6f}")

        # Sort by fitness descending
        population.sort(key=lambda x: x.fitness, reverse=True)
        best = population[0]
        best_fitness_history.append(best.fitness)
        print(f"Best fitness this generation: {best.fitness:.6f}")
        print(f"Best architecture: layers={best.n_layers}, units={best.units}, act={best.activation}, dropout={best.dropout:.3f}, lr={best.learning_rate}, l2={best.l2_reg}")

        # Elitism: keep top performers
        new_population = population[:keep_top]

        # Create offspring until population size is restored
        while len(new_population) < pop_size:
            parent1 = random.choice(population[:keep_top])
            parent2 = random.choice(population[:keep_top])
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    # Return best individual from final generation
    population.sort(key=lambda x: x.fitness, reverse=True)
    return population[0], best_fitness_history

# ---------------------------
# 3. Main Execution
# ---------------------------

def main():
    # Download stock data (e.g., Apple)
    ticker = "AAPL"
    print(f"Downloading data for {ticker}...")
    df = yf.download(ticker, start="2015-01-01", end="2024-01-01", interval="1d")
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    # Compute indicators
    print("Computing technical indicators...")
    df = compute_indicators(df)
    print(f"Data shape after indicators: {df.shape}")

    # Define feature columns and target
    feature_columns = ['RSI', 'EMA_9', 'EMA_21', 'EMA_50', 'BB_mid', 'BB_upper', 'BB_lower', 'BB_percent',
                       'MACD', 'MACD_signal', 'MACD_hist', 'ATR', 'VWAP', 'Return_1', 'Return_5']
    target_column = 'Close'

    # Create sequences
    window_size = 30
    future_steps = 1  # predict next candle close
    X, y = create_sequences(df, feature_columns, target_column, window_size, future_steps)
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Split data into train, validation, test (time-based, no shuffle)
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    # Scale features (fit only on training data to avoid leakage)
    print("Scaling features...")
    scaler = MinMaxScaler()
    # Reshape to 2D for scaling, then back to 3D
    orig_shape = X_train.shape
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_val_flat = X_val.reshape(-1, X_val.shape[-1])
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])

    X_train_flat = scaler.fit_transform(X_train_flat)
    X_val_flat = scaler.transform(X_val_flat)
    X_test_flat = scaler.transform(X_test_flat)

    X_train = X_train_flat.reshape(orig_shape)
    X_val = X_val_flat.reshape(X_val.shape)
    X_test = X_test_flat.reshape(X_test.shape)

    # Optionally scale target (y) as well
    y_scaler = MinMaxScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    # Run evolution
    print("\nStarting neuroevolution...")
    best_individual, history = evolve(X_train, y_train, X_val, y_val,
                                      pop_size=10, generations=15, keep_top=3)

    # Evaluate best model on test set
    print("\n--- Final Evaluation on Test Set ---")
    best_model = best_individual.build_model(X_train.shape[1:])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    best_model.fit(X_train, y_train,
                   validation_data=(X_val, y_val),
                   epochs=100, batch_size=32,
                   callbacks=[early_stop], verbose=1)

    test_pred = best_model.predict(X_test, verbose=0).flatten()
    test_rmse = np.sqrt(np.mean((test_pred - y_test) ** 2))
    print(f"Test RMSE (scaled): {test_rmse:.6f}")

    # Directional accuracy
    test_direction = np.sign(np.diff(y_test))
    pred_direction = np.sign(np.diff(test_pred))
    accuracy = np.mean(test_direction == pred_direction)
    print(f"Directional Accuracy: {accuracy*100:.2f}%")

    # Print final best architecture
    print("\nBest Individual:")
    print(f"Layers: {best_individual.n_layers}")
    print(f"Units: {best_individual.units}")
    print(f"Activation: {best_individual.activation}")
    print(f"Dropout: {best_individual.dropout:.3f}")
    print(f"Learning rate: {best_individual.learning_rate}")
    print(f"L2 regularization: {best_individual.l2_reg}")

if __name__ == "__main__":
    main()
