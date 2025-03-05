custom_prefix = """You will be provided with an original math problem and its intermediate thought, action and observation. Please generate a sub question that should utilize the results of the intermediate thought, action and observation to replace the related information of original problem and delete redundant irrelevant information. The sub question should be wrapped by <subquestion> and </subquestion>.
Here are some examples you may refer to:
"""

custom_suffix = """Now! It's your turn.

<original problem> 
{origin_problem}
</original problem> 
<intermediate thought> 
{thought}
</intermediate thought> 
<intermediate action>
{action}
</intermediate action> 
<intermediate observation> 
{observation}
</intermediate observation>"""

math_examples = [
    # example 1
    """<original problem> 
Let $f(x)=x^3+3$ and $g(x) = 2x^2 + 2x +1$. What is $g(f(-2))$?
</original problem> 
<intermediate thought> 
To find $g(f(-2))$, we first need to calculate $f(-2)$ and then substitute this result into $g(x)$.
</intermediate thought> 
<intermediate action>
```python
f_x = lambda x: x**3 + 3
result_f = f_x(-2)
result_f
```
</intermediate action> 
<intermediate observation> 
-5
</intermediate observation>
<subquestion> 
what is $g(-5)$ if $g(x) = 2x^2 + 2x + 1$?
</subquestion>""", 
# example 2
"""<original problem> 
The perimeter of a rectangular garden is 60 feet. If the length of the field is twice the width, what is the area of the field, in square feet?
</original problem> 
<intermediate thought> 
he perimeter (P) of a rectangle is given by the formula $P = 2l + 2w$, where $l$ is the length and $w$ is the width of the rectangle. We are given that the perimeter is 60 feet and the length is twice the width, which can be expressed as $l = 2w$. We need to find the area (A) of the rectangle, which is given by the formula $A = l \\times w$.
</intermediate thought> 
<intermediate action>
```python
from sympy import symbols, Eq, solve

# Define symbols
l, w = symbols('l w')

# Given perimeter
P = 60

# Length is twice the width
length_eq = Eq(l, 2*w)

# Perimeter equation
perimeter_eq = Eq(2*l + 2*w, P)

# Solve the system of equations for l and w
solution = solve((length_eq, perimeter_eq), (l, w))
print(solution)
```
</intermediate action> 
<intermediate observation> 
{l: 20, w: 10}
</intermediate observation>
<subquestion> 
Given that the length of a rectangle is 20 feet and the width is 10 feet, what is the area of the rectangle, in square feet?
</subquestion>"""]

gsm8k_examples = [
    # example 1
    """<original problem> 
Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a budget of $200 and spent $30 on a button-up shirt, $46 on suit pants, $38 on a suit coat, $11 on socks, and $18 on a belt. She also purchased a pair of shoes, but lost the receipt for them. She has $16 left from her budget. How much did Alexis pay for the shoes?
</original problem> 
<intermediate thought> 
To find out how much Alexis paid for the shoes, I need to calculate the total amount spent on the other items she bought and subtract this from her initial budget. The initial budget is $200, and she has $16 left after all her purchases. The equation for this calculation is:\n\n\\[ \\text{Cost of shoes} = \\text{Initial budget} - (\\text{Cost of shirt} + \\text{Cost of pants} + \\text{Cost of coat} + \\text{Cost of socks} + \\text{Cost of belt} + \\text{Money left}) \\]
</intermediate thought> 
<intermediate action>
```python
initial_budget = 200
cost_shirt = 30
cost_pants = 46
cost_coat = 38
cost_socks = 11
cost_belt = 18
money_left = 16

other_items = cost_shirt + cost_pants + cost_coat + cost_socks + cost_belt + money_left
print(other_items)
```
</intermediate action> 
<intermediate observation> 
143
</intermediate observation>
<subquestion> 
Alexis had a budget of $200 for her business attire and after purchasing several items, she was left with $16. If the total cost of the other items was $143, how much did she spend on the pair of shoes?
</subquestion>""", 
# example 2
"""<original problem> 
James creates a media empire.  He creates a movie for $2000.  Each DVD cost $6 to make.  He sells it for 2.5 times that much.  He sells 500 movies a day for 5 days a week.  How much profit does he make in 20 weeks?
</original problem> 
<intermediate thought> 
To calculate the profit James makes, we need to consider the initial cost of creating the movie, the cost of producing each DVD, the selling price of each DVD, the number of DVDs sold per day, the number of days he sells per week, and the total number of weeks. The profit for each DVD sold is the selling price minus the cost to make it. The total profit is the sum of the profit from each DVD sold minus the initial cost of creating the movie.\n\nThe profit per DVD can be calculated as follows:\nProfit per DVD = Selling price per DVD - Cost to make per DVD\n\nThe total profit can be calculated as follows:\nTotal profit = (Profit per DVD * Number of DVDs sold per day * Number of days per week * Number of weeks) - Initial cost of creating the movie\n\nLet's calculate the profit per DVD first.
</intermediate thought> 
<intermediate action>
```python
cost_per_dvd = 6
selling_price_per_dvd = 2.5 * cost_per_dvd
profit_per_dvd = selling_price_per_dvd - cost_per_dvd
print(profit_per_dvd)
```
</intermediate action> 
<intermediate observation> 
9.0
</intermediate observation>
<subquestion> 
Given that James makes a profit of $9.0 per DVD, sells 500 DVDs a day for 5 days a week, and the initial cost of creating the movie was $2000, how much total profit does he make in 20 weeks?
</subquestion>"""]

