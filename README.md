**Quick Start**

1. Install env according to SETUP.md
2. To train go to train&test.ipynb and run through first half training portion
3. To test load model you want to test in second half of train&test.ipynb, load all cells, and then run the final cell to display testing matrices. 
4. To use in real life run run.py and watching the following video to understand how it works: 

**Demo Video/Video Links**

https://youtube.com/shorts/BczC19fXey4?feature=share

This video shows the model working on a phone game.  

**Project Goals & Research Question**

In blackjack, card counting is a method of tracking the number of high cards (e.g., 9 and 10) versus low cards (e.g., 2 and 3) that have been dealt. Players have used card counting to gain an edge over casinos and make money. It is common for a card counter’s edge (edge = total money won / total money wagered) to be between 0.5% and 1.5%.
In my project, I aim to achieve a greater edge using Deep Q Reinforcement Learning (DQRL) and a feature vector that includes both a Hi-Lo card counting scheme and a probability distribution of the remaining cards to be drawn. My goal is to use a perfect memory of the remaining card distribution, along with card counting, to create a model that is better at making money playing blackjack than typical card counters.

**What It Does**

My model uses DQRL to train on a feature vector consisting of:
* A probability distribution of each remaining card value being drawn based on the cards already seen
* A Hi-Lo count, defined as the true count divided by the number of decks remaining. The true count is calculated as the sum of +1 for all cards valued 2–6 and -1 for 10s and Aces observed so far
* An ace flag, since aces can take on multiple values in blackjack (used only in the action model)
* The stake, or the amount the player has bet on the hand (used only in the action model)
* The dealer’s total (best possible score based on their hand) (used only in the action model)
* The player’s total (best possible score based on their hand) (used only in the action model)

Using this feature vector, the model first determines the optimal bet for a new game of blackjack based on previously seen cards. Then, it determines the optimal action (e.g., stand, hit, double) to maximize returns or make the player money wile playing Black Jack.

**Approaches**

To accomplish this project, I initially created a single MLP that took in the feature vector and output eight classifications (four betting actions and four gameplay actions). My rationale was that this would require only one replay buffer, one epsilon schedule, and one model to train.
Below is the model architecture:
![One Model Approach.jpg](Performance%20Visualization/Two%20Model%20Approach.jpg)!
Below is how this model performed when tested: 
![Old Test.png](Performance%20Visualization/Old%20Test.png)
As shown, the average return per game was negative (-0.02 unites per game). This indicates that the model did not learn to count cards effectively. By combining two separate dynamics (betting and gameplay decisions) into a single model, I believe the model used much of its capacity learning the rules of the game rather than optimizing performance through card counting.
To address this, I developed a model with two separate MLPs. The first MLP takes in only the probability distribution and card count to determine the wager amount. The second MLP takes in the full feature vector described earlier to decide the optimal action.

Below is the updated model architecture:
![Two Model Approach.jpg](Performance%20Visualization/Two%20Model%20Approach.jpg)
Below is how the model performed: 
![Fin Test.png](Performance%20Visualization/Fin%20Test.png)

As shown, the model generated positive returns over a large number of games (approx 0.32 units per game). This indicates that the updated architecture is both profitable and an improvement over the original approach. This is reinforced by the fact that the two-model architecture had a higher average return (0.32) than the one model architecture (-0.02). Both models were trained on six million episodes with the exact same hyperparameters. 
In conclusion, separating the models yielded the best performance. This approach allowed for different epsilon decay schedules, which was beneficial because the action model was easier to train. Allowing it to decay faster provided the betting model with a more stable and accurate target. Additionally, using different learning rates helped both models converge to a strong expected return.

**Evaluation**
Below are the training and testing metrics from my final model: 
![Fin Train.png](Performance%20Visualization/Fin%20Train.png)
![Fin Test.png](Performance%20Visualization/Fin%20Test.png)
As shown in the training metrics (top-right graph: Average Reward History), the model converges toward a positive average reward. However, due to the inherent randomness in blackjack, it is difficult to precisely determine the true expected return from this alone.
To evaluate the action model, the key metrics are the percentages of wins, draws, and losses. Under optimal play, a player wins approximately 45% of the time, draws about 10% of the time, and loses around 48%. As shown in the Normalized Outcomes Per Run graph, the model’s results align closely with these optimal benchmarks.
To evaluate the betting model, the Bet vs. True Count and Normalized Results Per Run graphs provide insight into how often different bet sizes are used and at which true counts. However, the most important metric is the edge. As shown in the test metrics, the model achieved an edge of approximately 4.3%. This suggests that the model is performing extremely well, significantly exceeding the typical player edge of 0.5–1.5%.
Overall, both the betting and action models demonstrate strong performance and successfully learn strategies consistent with - and with respect to the reward model better than - effective card counting. 4.3% edge is a surprising result as it is significantly better than standard card counting. However, the inclusion of the probability distribution is driving force behind the improvement. People do not have the capacity to memorize seen cards and their distributions exactly, thus the model has a sizeable advantage for predicting cards. Additionally, the environment I created is friendly for card counting. 0.87 penetration means that this is how much of the deck is played until they shuffle. 0.87 is a large percentage of the deck that the model gets to see. Thus, the farther into the deck the game goes the better off the model is (as it can accurately predict what is next based off of the cards that it has seen so far). Under simular conditions, a human card counter is estimates to get a 2.0% edge. Thus, even though the environment is tailored to card counting, the model still performs better than humans.  



