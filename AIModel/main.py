"""
End-to-end run: train the hybrid CNN+LSTM+MLP model, then evaluate on the
held-out subject-wise test set.

Usage:
    python main.py
"""
from train import main as train_main
from evaluate import evaluate_model


if __name__ == "__main__":
    model, (Xs_test, Xf_test, y_test, g_test), device = train_main()
    print(f"\nTest volunteers ({len(set(g_test))}): {sorted(set(g_test))}")
    evaluate_model(Xs_test, Xf_test, y_test, device=device)