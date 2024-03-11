# Strategies Against Each Opponent in Rock, Paper, Scissors

The `RPSGame` class is designed to outsmart different opponents in a Rock, Paper, Scissors game by employing tailored strategies. Below, we analyze each opponent's strategy and dissect the counter-strategies employed by `RPSGame`, explaining the Strategy behind each move and how it leads to desired results.

## Quincy's Pattern

Quincy follows a predictable pattern: `['Rock', 'Paper', 'Paper', 'Scissors', 'Rock']`. Our goal is to exploit this predictability by countering each move in Quincy's sequence.

### Game Strategy Against Quincy:

#### Round 1: Quincy plays 'Rock'
- **Action**: We anticipate Quincy's 'Rock' and play 'Paper' to win.
- **Strategy**: Knowing Quincy's start with 'Rock', 'Paper' is the ideal counter.

#### Round 2: Quincy plays 'Paper'
- **Action**: We counter with 'Scissors', anticipating Quincy's 'Paper'.
- **Strategy**: Sequential pattern knowledge allows us to counter effectively.

#### Round 3: Quincy repeats 'Paper'
- **Action**: We stick with 'Scissors', winning again.
- **Strategy**: Quincy's pattern is predictable; we exploit it with the perfect counter.

#### Round 4: Quincy switches to 'Scissors'
- **Action**: We counter with 'Rock'.
- **Strategy**: Anticipating the cycle, we play the counter to Quincy's 'Scissors'.

#### Round 5: Quincy goes back to 'Rock'
- **Action**: We loop back to playing 'Paper'.
- **Strategy**: The cycle repeats, and we counter Quincy's 'Rock' with 'Paper'.

By exploiting Quincy's cyclic pattern, we effectively counter each move, leading to consistent victories.

## Abbey's Adaptation

Abbey adapts its strategy based on analyzing our recent moves. We must be unpredictable and strategically diverse to outmaneuver Abbey.

### Game Strategy Against Abbey:

#### Early Rounds
- **Action**: Start with 'Paper', an unconventional choice aimed at throwing Abbey off.
- **Strategy**: 'Paper' is less predictable and might counter Abbey's expected 'Rock'.

#### Subsequent Rounds
- If Abbey adapts to our 'Paper' by playing 'Scissors', we switch to 'Rock' in the next round to counter Abbey's 'Scissors'.
- **Strategy**: We adapt to Abbey's adaptation, staying a step ahead.

#### Every Fifth Round
- We introduce a curveball by countering the counter to our last move.
- **Strategy**: This unpredictability disrupts Abbey's pattern analysis, making it harder for Abbey to predict Actions.

By being adaptive and occasionally unpredictable, we aim to outsmart Abbey's analytical strategy.

## Kris's Direct Counter

Kris plays a direct counter to our last move, which allows us to manipulate Kris's responses to our advantage.

### Game Strategy Against Kris:

#### Round 1: We play 'Scissors'
- Kris counters with 'Rock'.
- **Our Next Move**: Predicting Kris's 'Rock', we play 'Paper'.
- **Strategy**: We exploit Kris's predictability by always staying one step ahead.

#### Subsequent Rounds
- We continue to manipulate Kris by playing the move that Kris would counter based on our last move.
- **Strategy**: By understanding Kris's strategy, we control the game's flow, ensuring victories.

## Mrugesh's Frequency Analysis

Mrugesh counters the most frequent play in our last ten moves, requiring us to be strategic about Action diversity.

### Game Strategy Against Mrugesh:

#### Rounds 1-10
- We vary Actions, ensuring no single move dominates our play pattern.
- **Strategy**: Preventing any move from becoming "most frequent" hinders Mrugesh's strategy.

#### Adapting to Mrugesh
- If a pattern does emerge, we preemptively counter Mrugesh's expected counter.
- **Strategy**: By staying unpredictable and adjusting Actions, we neutralize Mrugesh's strategy.

# Conclusion

The key to success in `RPSGame` lies in understanding each opponent's strategy and tailoring our counter-strategy accordingly. By employing a mix of predictability, adaptability, and unpredictability at strategic moments, we navigate through opponents' patterns and adaptations, securing victories through calculated moves and counter-moves.
