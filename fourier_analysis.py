"""Fourier analysis of stock price data.

Fetches historical data from yfinance, computes the FFT of the closing prices,
identifies the top five frequencies, and plots both the actual prices and the
Fourier reconstruction extended into the future.
"""
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fourier_stock_analysis(ticker: str, period: str, interval: str, extension_days: int) -> None:
    """Fetch stock data, perform FFT, and plot reconstruction.

    Parameters
    ----------
    ticker: str
        Stock ticker symbol.
    period: str
        Period of historical data to download (e.g., "1y", "2y").
    interval: str
        Data interval (e.g., "1d", "1h").
    extension_days: int
        Number of days to extend the Fourier reconstruction for prediction.
    """
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    close = data["Close"].dropna()
    if close.empty:
        raise ValueError("No close price data retrieved. Check ticker and parameters.")

    n = len(close)
    t = np.arange(n)
    # Remove mean for FFT
    detrended = close - close.mean()
    fft = np.fft.fft(detrended)
    freqs = np.fft.fftfreq(n, d=1)

    # Only consider positive frequencies
    positive = freqs > 0
    freqs = freqs[positive]
    fft = fft[positive]

    # Find top five frequency components
    indices = np.argsort(np.abs(fft))[-5:]
    top_freqs = freqs[indices]
    top_coeffs = fft[indices]

    # Prepare time axis for reconstruction and prediction
    t_extended = np.arange(n + extension_days)
    reconstruction = np.zeros_like(t_extended, dtype=float)

    # Reconstruct signal using top frequencies
    for freq, coeff in zip(top_freqs, top_coeffs):
        amplitude = 2 * np.abs(coeff) / n
        phase = np.angle(coeff)
        reconstruction += amplitude * np.cos(2 * np.pi * freq * t_extended + phase)

    reconstruction += close.mean()

    # Plot actual data and reconstruction with prediction
    plt.figure(figsize=(12, 6))
    plt.plot(t, close.values, label="Actual")
    plt.plot(t_extended, reconstruction, label="Fourier approx + prediction")
    plt.axvline(n - 1, color="red", linestyle="--", label="Prediction start")
    plt.title(f"{ticker} Closing Price and Fourier Approximation")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    output_file = f"{ticker}_fourier.png"
    plt.savefig(output_file)
    plt.close()
    print(f"Plot saved to {output_file}")

    # Print frequency information
    periods = 1 / np.abs(top_freqs)
    info = pd.DataFrame({"frequency (1/day)": top_freqs, "period (days)": periods})
    info = info.sort_values("frequency (1/day)")
    print("Top 5 frequencies:")
    print(info)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fourier analysis of stock prices.")
    parser.add_argument("ticker", nargs="?", default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--period", default="1y", help="Data period, e.g., 1y, 2y")
    parser.add_argument("--interval", default="1d", help="Data interval, e.g., 1d, 1h")
    parser.add_argument("--extend", type=int, default=30, help="Days to predict")
    args = parser.parse_args()
    fourier_stock_analysis(args.ticker, args.period, args.interval, args.extend)


if __name__ == "__main__":
    main()
