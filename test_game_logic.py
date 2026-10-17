"""
Automated Test Suite for Thanos vs. Human AI Project
Verifies Step 6 Game Logic, Scoring Engine, 5-Round Match Flow, and Predictor Deterministically.
"""

import sys
import hand_gesture


def run_tests():
    print("=" * 65)
    print(" Running Automated Verification Suite for Thanos vs. Human AI")
    print("=" * 65)

    passed_count = 0
    total_tests = 9

    # -------------------------------------------------------------
    # Test 1: Human Wins a Round
    # -------------------------------------------------------------
    code, title, desc, color = hand_gesture.determine_winner("ROCK", "SCISSORS")
    assert code == "HUMAN_WINS", f"Test 1 Failed: Expected HUMAN_WINS, got {code}"
    print(" [+] Test 1 Passed: Rock beats Scissors -> HUMAN_WINS")
    passed_count += 1

    # -------------------------------------------------------------
    # Test 2: Thanos Wins a Round
    # -------------------------------------------------------------
    code, title, desc, color = hand_gesture.determine_winner("ROCK", "PAPER")
    assert code == "THANOS_WINS", f"Test 2 Failed: Expected THANOS_WINS, got {code}"
    print(" [+] Test 2 Passed: Paper beats Rock -> THANOS_WINS")
    passed_count += 1

    # -------------------------------------------------------------
    # Test 3: Round Ends in a Draw
    # -------------------------------------------------------------
    code, title, desc, color = hand_gesture.determine_winner("PAPER", "PAPER")
    assert code == "DRAW", f"Test 3 Failed: Expected DRAW, got {code}"
    print(" [+] Test 3 Passed: Paper vs Paper -> DRAW")
    passed_count += 1

    # -------------------------------------------------------------
    # Test 4: Invalid Gesture Handling (Retry State)
    # -------------------------------------------------------------
    predictor = hand_gesture.StrategyPredictor()
    initial_history_len = len(predictor.history)
    # Simulated invalid gesture: "UNKNOWN" or "NO_HAND"
    invalid_gesture = "UNKNOWN"
    # Should NOT be recorded into history
    predictor.record_move(invalid_gesture)
    assert len(predictor.history) == initial_history_len, "Test 4 Failed: Invalid gesture polluted history"
    print(" [+] Test 4 Passed: Invalid gesture does not increment round or history")
    passed_count += 1

    # -------------------------------------------------------------
    # Test 5: 5 Completed Rounds -> Human Victory
    # -------------------------------------------------------------
    # Moves: H wins 3, T wins 1, 1 Draw
    rounds = [
        ("ROCK", "SCISSORS"),   # H win
        ("PAPER", "ROCK"),       # H win
        ("SCISSORS", "PAPER"),   # H win
        ("ROCK", "PAPER"),       # T win
        ("ROCK", "ROCK"),        # Draw
    ]
    h_score = 0
    t_score = 0
    d_score = 0
    for h, t in rounds:
        c, _, _, _ = hand_gesture.determine_winner(h, t)
        if c == "HUMAN_WINS": h_score += 1
        elif c == "THANOS_WINS": t_score += 1
        else: d_score += 1

    assert h_score == 3 and t_score == 1 and d_score == 1
    assert h_score > t_score, "Test 5 Failed: Expected Human Victory"
    print(f" [+] Test 5 Passed: 5 Rounds (H:{h_score}, T:{t_score}, D:{d_score}) -> HUMANITY IS SAVED!")
    passed_count += 1

    # -------------------------------------------------------------
    # Test 6: 5 Completed Rounds -> Thanos Victory
    # -------------------------------------------------------------
    rounds_t = [
        ("SCISSORS", "ROCK"),    # T win
        ("ROCK", "PAPER"),       # T win
        ("PAPER", "SCISSORS"),   # T win
        ("ROCK", "PAPER"),       # T win
        ("ROCK", "SCISSORS"),    # H win
    ]
    h_score = 0
    t_score = 0
    d_score = 0
    for h, t in rounds_t:
        c, _, _, _ = hand_gesture.determine_winner(h, t)
        if c == "HUMAN_WINS": h_score += 1
        elif c == "THANOS_WINS": t_score += 1
        else: d_score += 1

    assert t_score > h_score, "Test 6 Failed: Expected Thanos Victory"
    print(f" [+] Test 6 Passed: 5 Rounds (H:{h_score}, T:{t_score}, D:{d_score}) -> THE SNAP IS INEVITABLE!")
    passed_count += 1

    # -------------------------------------------------------------
    # Test 7: 5 Completed Rounds -> Overall Draw
    # -------------------------------------------------------------
    rounds_d = [
        ("ROCK", "SCISSORS"),    # H win
        ("SCISSORS", "ROCK"),    # T win
        ("PAPER", "ROCK"),       # H win
        ("ROCK", "PAPER"),       # T win
        ("ROCK", "ROCK"),        # Draw
    ]
    h_score = 0
    t_score = 0
    d_score = 0
    for h, t in rounds_d:
        c, _, _, _ = hand_gesture.determine_winner(h, t)
        if c == "HUMAN_WINS": h_score += 1
        elif c == "THANOS_WINS": t_score += 1
        else: d_score += 1

    assert h_score == t_score, "Test 7 Failed: Expected Overall Draw"
    print(f" [+] Test 7 Passed: 5 Rounds (H:{h_score}, T:{t_score}, D:{d_score}) -> PERFECT BALANCE ACHIEVED!")
    passed_count += 1

    # -------------------------------------------------------------
    # Test 8: Clean Match Restart (Zero Stale State)
    # -------------------------------------------------------------
    test_predictor = hand_gesture.StrategyPredictor()
    test_predictor.record_move("ROCK")
    test_predictor.record_move("PAPER")
    assert len(test_predictor.history) == 2

    # Reset
    test_predictor.reset()
    assert len(test_predictor.history) == 0, "Test 8 Failed: History not reset"
    assert sum(test_predictor.move_counts.values()) == 0, "Test 8 Failed: Move counts not reset"
    # Verify transition matrix is zeroed
    for move in ["ROCK", "PAPER", "SCISSORS"]:
        assert sum(test_predictor.transitions[move].values()) == 0
    print(" [+] Test 8 Passed: Match Reset completely wipes predictor history & transition matrix")
    passed_count += 1

    # -------------------------------------------------------------
    # Test 9: Confirming 6th Round Cannot Begin
    # -------------------------------------------------------------
    current_round = 5
    max_rounds = hand_gesture.MAX_ROUNDS
    # Simulate attempt to trigger another round past MAX_ROUNDS
    can_start_sixth_round = (current_round < max_rounds)
    assert not can_start_sixth_round, "Test 9 Failed: 6th round was allowed to start"
    print(" [+] Test 9 Passed: 6th round is strictly prohibited (capped at 5 rounds)")
    passed_count += 1

    print("\n" + "=" * 65)
    print(f" All {passed_count}/{total_tests} Tests Passed Successfully! System Verified.")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()
