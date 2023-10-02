## LLM Operator framework 
Create your own operator! 

Build an LLM operator to intelligently plan, select, and invoke different functions in your application, ie. functions, APIs, or tools to use.

```
foodOperator = FoodDeliveryOperator()
foodOperator.add_operation(foodOperator.search)
foodOperator.add_operation(foodOperator.order)
foodOperator.train(<training_file>, <operator_save_path>)
response = foodOperator("I want 10l of milk.")
```

### Framework

`Operator` - the main entity(class) that encapsulates similar operations together.
Eg: `OnboardingOperator` which has operations to understand and save user information like name, email, age, etc.
`FoodDeliveryOperator` which has operations like search something about the app, ask a general query or place an order.
The framework intelligently decides which operation to call and the required arguments from the user input.

`Operation` - functions within your operator class which carry the business logic. Multiple operations reside within an operator.
Eg: setAge, setEmailAddress, setHeight.

You can also allow chat through your operator by defining a chat operation. Here you can pass your own fine-tuned LLM model to chat with the user. 
Eg: `FoodDeliveryOperator` in `test_food_delivery.py` instantiates a chat LLM to chat with the user. Operation `noop` is invoked when a general query is detected. This operation calls the chat LLM to send an appropriate response to the user.

#### How to create an Operator?
Here's an example. You are building an application with a chat-based onboarding flow that gathers information about the user's demographic information, e.g. age and height, as your LLM has a conversation with the user.

First create an `OnboardingOperator` by extending the `Operator` class:
```
class OnboardingOperator(Operator):
```

Create a `setAge` method inside the class. This is an example of an operation that the `OnboardingOperator` can invoke. For example, a user might say `who me? I am of age fifty nine, my friend.` Given this, you may want to initiate an operation to extract the age from this message and say, store it in a database. So, you would expect the Operator to call a function `setAge` that extracts the correct age `{'age': 59}` and then stores it in a database.

To make this understandable to the `OnboardingOperator`, a natural language description can be prompt-engineered to explain what it does, e.g. `set the age of a person`. Define any other business logic inside this function.

Also, add prompt-engineered descriptions to the parameters to provide contextual information to the LLM Operator:
```
def setAge(self, age: int):
    """
    set the age of a person
    
    Parameters:
    age: age of the person in years.
    """
    ### Do whatever you want here, e.g. save the age to database, do some analysis, etc.
    ### As a hello world example, this is returning a string with the extracted age parameter
    return f"Hello! Your age has been set to {age}"
```

You can add many more operations like `setHeight`, `setEmailAddress`, etc. to the `OnboardingOperator` class and train the operator to distinguish between them using the docstrings of the functions.
Additionally, you can also train an operator to call other operators which in turn call the desired operations!
For example, see `test_main.py` which has a `MainAppOperator` that can call `OnboardingOperator` or `MotivationOperator` based on user input!

![fullApp.png](images%2FfullApp.png)

Build a chain of operators and define a flow of your application.

![chat.png](images%2Fchat.png)

### Steps:

1. Create an operator class. Examples in `test_onboarding.py`, `test_motivation.py` and `test_main.py`. 
2. Create operations within the Operator to define the tasks you want to do. Follow the docstring format for each function to specify the description of the operation and each parameter within it.
3. Add all your desired operations using `operator.add_operation(<operation_callback>)`.
4. Train your operator using the docstrings inside each operation to clarify their purpose. Additionally, you can also train it with some labelled examples like in `train_clf.csv`. This is recommended for accuracy. 

    Train using `operator.train(<optional_training_file_path>, <operator_save_path>)`.
5. After training, you can load your trained operator using something like `operator = OnboardingOperator().load(<operator_save_path>)`.
6. Now, you can start using your operator for routing between operations and executing the right one using `response = operator(<query>)`.

### How to recreate and run your operator
1. Download requirements using `pip3 install -r requirements.txt`.
2. You can create an operator class like in `examples/test_food_delivery.py`.
3. You can change the operator(class) name, operations(functions) and their descriptions as per your use case. Define your own business logic within each operation.
4. Check `main()` to see how to execute your operator framework.
5. Run the file using `python3 examples/test_food_delivery.py`.

### Examples

#### Onboarding Operator example

Code:
```
operator_save_path = "examples/models/clf/OnboardingOperator/router.pkl"
operator = OnboardingOperator().load(operator_save_path)
operator.add_operations()
query2 = "who me? I am of age fifty nine, my friend."
print(f"\n\User input: {query2}")
response2 = operator(query2)
print(response2)
```

Output:
```
User input: who me? I am of age fifty nine, my friend.

Selected operation: setAge
Inferred arguments: {'age': '59'}

It is indicated to be the age of the user.
Age has been set. Age= 59
```

#### A Food Delivery Operator example

Code:
```
operator_save_path = "examples/models/clf/FoodDeliveryOperator/router.pkl"
foodOperator = FoodDeliveryOperator().load(operator_save_path)
foodOperator.add_operations()
query1 = "I want 10l of milk."
print(f"\n\nUser input: {query1}")
response1 = foodOperator(query1)
print(response1)
```
Output:
```
User input: I want 10l of milk.

Selected operation: order
Inferred arguments: {'item_name': 'milk', 'quantity': '10', 'unit': 'liters'}

It is indicated that the user wants to place an order.
Calling orders API with: item_name=milk, quantity=10, unit=liters

```

#### A Food Delivery chat example

Code:
```
operator_save_path = "examples/models/clf/FoodDeliveryOperator/router.pkl"
foodOperator = FoodDeliveryOperator().load(operator_save_path)
foodOperator.add_operations()
query3 = "Are there any exercises I can do to lose weight?"
print(f"\n\nQuery2: {query3}")
response3 = foodOperator(query3)
print(response3)
```
Output:
```
User input: Are there any exercises I can do to lose weight?

Selected operation: noop
Inferred arguments: {'message': 'Are there any exercises I can do to lose weight?'}

It is indicated that this is a general query. So redirecting to a chat LLM.
Calling general query LLM...
user query= Are there any exercises I can do to lose weight? 
output= Yes, there are many exercises that can help you lose weight. Cardiovascular exercises such as running, cycling, and swimming are effective for burning calories and improving cardiovascular health. Resistance training, such as weightlifting or bodyweight exercises, can also help build muscle mass, which can increase your metabolism and help you lose weight.

```
