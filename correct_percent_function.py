# function that calculates the percentage of correct indications
def calculate_percent_correct(correct, wrong):
    sum_all = correct + wrong
    if sum_all == 0:
        return 0 # Prevents division by zero
    percent_correct = (correct / sum_all) * 100
    return percent_correct