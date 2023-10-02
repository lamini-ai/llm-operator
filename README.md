# LLM Operators - Build custom planning & tool-using LLMs with [Lamini](https://lamini.ai)

Build your own Operator! An Operator is an LLM that can intelligently plan, select, and invoke different functions in your application. Here's a toy example of a [food delivery app](examples/test_food_delivery.py) which handles some operations like search or place an order:

```
food_operator = FoodDeliveryOperator()

food_operator.add_operation(food_operator.search)
food_operator.add_operation(food_operator.order)

food_operator.train(<training_file>, <operator_save_path>)
response = food_operator("I want 10l of milk.")
```

You can run it directly like this:
```bash
python examples/test_food_delivery.py
```
Notes:
* Install the requirements first: `pip install -r requirements.txt`.
* Include this repo in your python path: `export PYTHONPATH=$PYTHONPATH:./`.

Here is the Food Delivery Operator's thought process and plan:
```
Query: Add 10 gallons of milk to my cart.

selected operation: order
inferred arguments: {'item_name': 'milk', 'quantity': '10', 'unit': 'gallons'}

It is indicated that the user wants to invoke cart/order operation.
Calling orders API with: item_name=milk, quantity=10, unit=gallons
```

### Chat x Operator
LLM Operators can work hand in hand with your other LLMs, e.g. for Q&A, chat, etc.:
```
self.chat = LlamaV2Runner() # inside Operator class, pass in a model_name to a finetuned LLM if desired
...
message = user_input + orders_api_response
model_response = self.chat(message, system_prompt=f"Respond to the user, confirming the addition to cart. If response from API is 200, then confirm that the item {item_name} has been placed in the cart, else ask the user to restate their order.")
```

See [`FoodDeliveryOperator`](examples/test_food_delivery.py) for a complete example.

## Create Your Own Operator

tl;dr:
* Create an Operator class with operations (functions) for it to invoke. Lots of examples [here](examples/) :)
* Finetune your Operator with prompt-engineered descriptions and/or data - now it can intelligently invoke operations! All of the `examples` operators are pre-trained for you to try immediately. The trained operator is saved as `router.pkl` in the respective operator folder. :)
* Hook your custom Operator LLM up to your own application with a simple [REST API](https://lamini-ai.github.io/API/completions/).

1. Create an operator class. Examples:
    * [`test_onboarding.py`](examples/test_onboarding.py): onboards users, extracting demographic data
    * [`test_motivation.py`](examples/test_motivation.py): motivates, reminds, and follows up with users
    * [`test_food_delivery.py`](examples/test_food_delivery.py): orders or searches for users in a food delivery app
    * [`test_main.py`](examples/test_main.py): **Advanced**, combines the onboarding and motivation operators together in a larger app

2. Create operations (functions) that you want the Operator to invoke. Here is the `order` operation for ordering food: 
```
def order(self, item_name: str, quantity: str, unit: str):
   """
   User wants to order items, i.e. buy an item and place it into the cart. This includes any indication of wanting to checkout, or add to or modify the cart. It includes mentioning specific items, or general turn of phrase like 'I want to buy something'.
   
   Parameters:
   item_name: name of the item that the user wants to order.
   quantity: quantity of the item that the user wants to order.
   unit: unit of the item that the user wants to order like kilograms, pounds, etc.
   """
```
You can prompt-engineer the docstring! The main docstring and parameter descriptions are all read by the LLM Operator to follow your instructions. This will help your Operator learn the difference between operations and what parameters it needs to extract for each operation.

3. In the Operator's main call function, register each of your operations, e.g.:
```
operator.add_operation(self.order)
operator.add_operation(self.search)
...
```

4. Finetune your Operator! For best results, give it some examples like in [`train_clf.csv`](examples/models/clf/FoodDeliveryOperator/train_clf.csv) for `FoodDeliveryOperator`. Finetuning is a form of training. We suggest giving atleast 50 examples per operation. The more, the better!
```
optional_training_filepath = "examples/models/clf/FoodDeliveryOperator/train_clf.csv" # extra training data
operator_save_path = "examples/models/clf/FoodDeliveryOperator/router.pkl" # save to use later

operator.train(optional_training_filepath, operator_save_path)
```
Fun fact: `clf` stands for classifier, because your operator is actually classifying a user request into different operations!

5. Use your finetuned Operator — on as many user queries as you'd like!
```
finetuned_operator = FoodDeliveryOperator().load(operator_save_path)

user_query = "Add 10 gallons of milk to my cart."
response = finetuned_operator(user_query)
```
Hook your custom LLM Operator up to your production application with a simple [REST API](https://lamini-ai.github.io/API/completions/) call.

## Operator Framework - super simple!

`Operator` - main class that intelligently plans which operation (function) to invoke, e.g.:
* [`OnboardingOperator`](examples/test_onboarding.py): calls operations to extract and save user information like name, email, age, etc.
* [`FoodDeliveryOperator`](examples/test_food_delivery.py): calls operations to search an FAQ or place an order.
* [`MotivationOperator`](examples/test_motivation.py): calls operations to send different types of messages to users to motivate, remind, or follow up with them.
* [`MainApp`](examples/test_main.py): **Advanced** main operator that calls the `OnboardingOperator` and `MotivatorOperator` as operations in a larger app. So yes, you can also train an operator to call other operators, which in turn call the operations you want it to call -- it's operators all the way down!

`Operation` - functions that your Operator can invoke. Multiple operations can reside within an Operator. For example: 
* [`OnboardingOperator`](examples/test_onboarding.py): setAge, setEmailAddress, setHeight.
* [`FoodDeliveryOperator`](examples/test_food_delivery.py): search, order, noop.
* [`MotivationOperator`](examples/test_motivation.py): setReminder, sendCongratsMessage, sendFollowupMessage.
These operations also include parameters that you want the Operator to extract in order to properly invoke these operations. For example, in `setAge`, the desired parameter would be `age` that can be extracted and then, for example, saved in a database about the user.

Connect your Chat LLM to your Operator. An example of this in the [`FoodDeliveryOperator`](examples/test_food_delivery.py) when no specific operation is selected, and the user is just chatting generally:
```
Query: Are there any exercises I can do to lose weight?
selected operation: noop

It is indicated that this is a general query. So redirecting to a general chat LLM.
Calling general chat LLM...
output=
   Yes, there are many exercises that can help you lose weight. Cardiovascular exercises such as running, cycling, and swimming are effective for burning calories and improving cardiovascular health. Resistance training, such as weightlifting or bodyweight exercises, can also help build muscle mass, which can increase your metabolism and help you lose weight.
```

You can, of course, customize this to your own finetuned chat LLMs.

