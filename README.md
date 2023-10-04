# LLM Operators - Build custom planning & tool-using LLMs with [Lamini](https://lamini.ai)

Build your own Operator! An Operator is an LLM that can intelligently plan, select, and invoke different functions in your application. Here's a toy example of a [food delivery app](llm_operator/food_delivery_operator.py) which handles some operations like search or place an order:

```
food_operator = FoodDeliveryOperator()

food_operator.add_operation(food_operator.search)
food_operator.add_operation(food_operator.order)

food_operator.train(<training_file>, <operator_save_path>)
response = food_operator("I want 10l of milk.")
```

You can run it directly like this:
```bash
./food_delivery_operator.sh
```

Here is the Food Delivery Operator's thought process and plan:
```
Query: Add 2 gallons of milk to my cart.

selected operation: order
inferred arguments: {'item_name': 'milk', 'quantity': '2', 'unit': 'gallons'}

It is indicated that the user wants to invoke cart/order operation.
Calling orders API with: item_name=milk, quantity=2, unit=gallons
```

You can also run your own queries like this:
```bash
./food_delivery_operator.sh --query "Add 2 gallons of milk to my cart." "I want 1 liter of milk."
```

### Chat x Operator
LLM Operators can work hand in hand with your other LLMs, e.g. for Q&A, chat, etc.:
```
self.chat = LlamaV2Runner() # inside Operator class, pass in a model_name to a finetuned LLM if desired
...
message = user_input + orders_api_response
model_response = self.chat(message, system_prompt=f"Respond to the user, confirming the addition to cart. If response from API is 200, then confirm that the item {item_name} has been placed in the cart, else ask the user to restate their order.")
```

To really supercharge your chat LLM, see our [`DocsToQA` SDK](https://github.com/lamini-ai/docs-to-qa) for how to prompt-engineer your way into a custom finetuned chat LLM on your raw documents.


### See the Operator Finetune in Action
This is how we trained the Operator above.
```bash
./food_delivery_operator.sh --train --operator_save_path models/AnotherFoodDeliveryOperator/ --training_data data/food_delivery.csv
```

We include ~30 [super simple datapoints](data/food_delivery.csv) to get a boost in performance, beyond just prompt-engineering. You can also use NO data, and just prompt-engineer -- or as we say "prompt-train"! A few rows of data:
| class_name | data                                               |
|------------|----------------------------------------------------|
| search     | "how do i track deliveries? substitutions?"       |
| order      | "I'd like to buy a bag of granny smith apples"    |
| noop       | "sometimes I dream of home"                        |


Run your finetuned Operator on your own queries, just as above:
```bash
./food_delivery_operator.sh --operator_save_path models/AnotherFoodDeliveryOperator/ --query "Add 2 gallons of milk to my cart, please" 
```

## Create Your Own Operator

tl;dr:
* Create an Operator class with operations (functions) for it to invoke. Lots of examples [here](llm_operator/) :)
* Finetune your Operator with prompt-engineered descriptions and/or data - now it can intelligently invoke operations! All of the `examples` operators are pre-trained for you to try immediately. The trained operator is saved as `router.pkl` in the respective operator folder. :)
* Hook your custom Operator LLM up to your own application with a simple [REST API](https://lamini-ai.github.io/API/completions/).

1. Create an operator class. Examples:
    * [`onboarding_operator.py`](llm_operator/onboarding_operator.py): onboards users, extracting demographic data
    * [`motivation_operator.py`](llm_operator/motivation_operator.py): motivates, reminds, and follows up with users
    * [`food_delivery_operator.py`](llm_operator/food_delivery_operator.py): orders or searches for users in a food delivery app
    * [`customer_support_operator.py`](llm_operator/customer_support_operator.py): handles customer support incl. opening, closing, and escalating tickets
    * [`operator_of_operators.py`](llm_operator/operator_of_operators.py): **Advanced**, combines the onboarding and motivation operators together in a larger app

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

3. In the Operator's main call function, register each of your operations in your class init function, e.g.:
```
operator.add_operation(self.order)
operator.add_operation(self.search)
...
```

4. Finetune your Operator! For best results, give it some examples like in [`food_delivery.csv`](data/food_delivery.csv) for `FoodDeliveryOperator`. Finetuning is a form of training. We suggest giving atleast 50 examples per operation. The more, the better!
```bash
./food_delivery_operator.sh --train --operator_save_path models/AnotherFoodDeliveryOperator/ --training_data data/food_delivery.csv
```

Or, in Python directly:
```
training_data = "data/food_delivery.csv" # extra training data (optional)
operator_save_path = "models/FoodDeliveryOperator/" # dirpath to save operator for use later

operator.train(training_data, operator_save_path)
```

The CSV data used is really simple and looks like [this](data/food_delivery.csv), with the correct `class_name` (operation name) and `data` (user query):
| class_name | data                                               |
|------------|----------------------------------------------------|
| search     | "how do i track deliveries? substitutions?"       |
| order      | "I'd like to buy a bag of granny smith apples"    |
| noop       | "sometimes I dream of home"                        |


5. Use your finetuned Operator — on as many user queries as you'd like!
```bash
./food_delivery_operator.sh --operator_save_path models/AnotherFoodDeliveryOperator/ --query "Add 2 gallons of milk to my cart, please" 
```

Or, in Python directly:
```
finetuned_operator = FoodDeliveryOperator().load(operator_save_path)

user_query = "Add 2 gallons of milk to my cart."
response = finetuned_operator(user_query)
```
Hook your custom LLM Operator up to your production application with a simple [REST API](https://lamini-ai.github.io/API/completions/) call.

## Operator Framework - super simple!

[`Operator`](llm_operator/base_operator.py) - main class that intelligently plans which operation (function) to invoke, e.g.:
* [`OnboardingOperator`](llm_operator/onboarding_operator.py): calls operations to extract and save user information like name, email, age, etc.
* [`FoodDeliveryOperator`](llm_operator/food_delivery_operator.py): calls operations to search an FAQ or place an order.
* [`MotivationOperator`](llm_operator/motivation_operator.py): calls operations to send different types of messages to users to motivate, remind, or follow up with them.
* [`CustomerSupportOperator.py`](llm_operator/customer_support_operator.py): calls operations to create tickets, resolve them, escalate them, and continue chatting with the user to gather more information.
* [`MainApp`](llm_operator/operator_of_operators.py): **Advanced** main operator that calls the `OnboardingOperator` and `MotivatorOperator` as operations in a larger app. So yes, you can also train an operator to call other operators, which in turn call the operations you want it to call -- it's operators all the way down!

`Operation` - functions that your Operator can invoke. Multiple operations can reside within an Operator. For example: 
* [`OnboardingOperator`](llm_operator/onboarding_operator.py): setAge, setEmailAddress, setHeight.
* [`FoodDeliveryOperator`](llm_operator/food_delivery_operator.py): search, order, noop.
* [`MotivationOperator`](llm_operator/motivation_operator.py): setReminder, sendCongratsMessage, sendFollowupMessage.
* [`CustomerSupportOperator.py`](llm_operator/customer_support_operator.py): create_ticket, close_ticket, escalate, gather_info.
* [`MainApp`](llm_operator/operator_of_operators.py): call_onboarding_operator, call_motivation_operator.

These operations also include parameters that you want the Operator to extract in order to properly invoke these operations. For example, in `setAge`, the desired parameter would be `age` that can be extracted and then, for example, saved in a database about the user.

Connect your Chat LLM to your Operator. An example of this in the [`FoodDeliveryOperator`](llm_operator/food_delivery_operator.py) when no specific operation is selected, and the user is just chatting generally:
```
Query: Are there any exercises I can do to lose weight?
selected operation: noop

It is indicated that this is a general query. So redirecting to a general chat LLM.
Calling general chat LLM...
output=
   Yes, there are many exercises that can help you lose weight. Cardiovascular exercises such as running, cycling, and swimming are effective for burning calories and improving cardiovascular health. Resistance training, such as weightlifting or bodyweight exercises, can also help build muscle mass, which can increase your metabolism and help you lose weight.
```

You can, of course, customize this to your own finetuned chat LLMs. See our [`DocsToQA` SDK](https://github.com/lamini-ai/docs-to-qa) for how to prompt-engineer your way into a custom finetuned chat LLM on your raw documents.

