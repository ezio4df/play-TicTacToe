# Tic-Tac-Toe Player

An intelligent Tic-Tac-Toe game where you play against an **unbeatable AI** powered by the **Minimax algorithm with alpha-beta pruning**. Built in Python, this project demonstrates foundational concepts in **game theory** and **classical AI**.
Perfect for students learning AI/ML, algorithms, or software development!

---

## Features

- Human vs. AI gameplay (you can't win—only draw or lose!)
- AI uses **optimal Minimax strategy** with **alpha-beta pruning** for efficiency
- Clean command-line interface
- Modular, readable, and well-commented code
- Includes game state validation and win/draw detection

---

## 🚀 Quick Start

### Prerequisites
- Python 3.6+

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/tic-tac-toe-ai.git
   cd tic-tac-toe-ai
   ```

2. Run the game:
   ```bash
   python tictactoe.py
   ```

### Gameplay
- The board is numbered 1–9 (like a phone keypad):
  ```
   1 | 2 | 3
  -----------
   4 | 5 | 6
  -----------
   7 | 8 | 9
  ```
- Enter a number (1–9) to place your mark (`X`).
- The AI plays as `O` and always makes the optimal move.

---

## 🧠 How It Works

- **Minimax Algorithm**: Recursively explores all possible future game states, assuming both players play optimally.
- **Alpha-Beta Pruning**: Eliminates branches that cannot influence the final decision, drastically improving performance.
- The AI evaluates terminal states with:
  - `+10` for AI win
  - `-10` for human win
  - `0` for a draw

This project illustrates how **deterministic, perfect-information games** can be solved exactly using search-based AI—without machine learning!

---

## 🛠️ Customization Ideas

- Add a GUI using `tkinter` or `pygame`
- Implement a "difficulty level" (e.g., random mistakes at lower levels)
- Extend to larger boards (e.g., 4x4)
- Log game states for analysis

---

## 📚 Learning Value

This project covers key CS/AI concepts:
- Recursion & game trees
- Adversarial search
- Algorithm optimization (pruning)
- Separation of concerns (game logic vs. AI logic)

Great for portfolios, interviews, or personal study!

---

## 📜 License

MIT License — feel free to use, modify, and share!

---

## 💬 Feedback?

Found a bug? Have an idea? Open an issue or PR!  
*(Replace with your contact info or social links if desired)*

---

> “In Tic-Tac-Toe, perfection is possible. In life—play to learn.” 🧠

---

Let me know if you used a different approach (e.g., Q-learning, web frontend with Flask/React, etc.), and I’ll tailor the README accordingly!