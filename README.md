## LLM Operator framework 
Create your own operator framework! 

Build an LLM framework to intelligently select different functions to perform in your application: from planning different operations to use (ie. functions, APIs, or tools to use) to invoking those operations.

For example, a user might say `who me? I am of age fifty nine, my friend.`. You want to extract the age from this message. So, you would expect the LLM to call a function `setAge` that extracts the correct age `{'age': 59}`.

#### Workflow: LLM Onboarding Operator
Here's an example. You are building an application with a chat-based onboarding flow that gathers information about the user's demographic information, e.g. age and height, as your LLM has a conversation with the user.

First create an `OnboardingOperator` by extending the `Operator` class:
```
class OnboardingOperator(Operator):
```

Create a `setAge` method inside the class. This is an example of an operation that the `OnboardingOperator` can invoke. For example, if a user sends a message like "I'm 36 years old", the `OnboardingOperator` can invoke the `setAge` method and extract the parameter `age` from the message.
```
def setAge(self, age: int):
```

To make this understandable to an LLM like `OnboardingOperator`, a natural language description can be prompt-engineered to explain what it does, e.g. `set the age of a person`.
```
def setAge(self, age: int):
    """
    set the age of a person
    """
```

Next, adding prompt-engineered descriptions to the parameters also provides contextual information to the LLM Operator:
```
def setAge(self, age: int):
    """
    set the age of a person
    
    Parameters:
    age: age of the person in years.
    """
```

Finally, once the operator routes to this function, you can do whatever you want in this function with the extracted parameter `age`. For example, you can save the age to a database, or you can return a customer message with the age, or you can call another LLM.
```
def setAge(self, age: int):
    """
    set the age of a person
    
    Parameters:
    age: age of the person in years.
    """
    ### Do whatever you want here, e.g. save the age to database, do some analysis, etc.
    ### As a hello world example, this is returning a string with the extract age parameter
    return f"Hello! Your age has been set to {age}"
```

Now, add additional operations! In our example, you can also see setting the height of the user. 

Finally, you can then train the Operator to route to the right operations. 
Additionally, you can also train an operator to call other operators which in turn call the desired operations! Build a chain of operations and define a flow of your application.
For example, see `test_main.py` which is an Operator that can call `OnboardingOperator` or `MotivationOperator` based on user input!
![fullApp.png](images%2FfullApp.png)

### Framework

`Operator` - the main entity that encapsulates similar operations together.
Eg: OnboardingOperator which has operations to understand and save user information like name, email, age, etc.

`Operation` - multiple operations reside within an operator. The operator calls these operations based on user inputs to do the desired operation.
Eg: setAge, setEmailAddress, setHeight.

The framework intelligently decides which operation to call and the required arguments from the user input.

### Steps:

1. Create your operator class. Examples in `test_onboarding.py`, `test_motivation.py` and `test_main.py`. Follow the docstring format for each function to specify the description of the operation and each parameter within it.
2. Add all your desired operations using `operator.add_operation(<operation_callback>)`.
3. For the very first time you would have to train your operator to give examples on what user operation should be invoked for what user query. You can train using `operator.train()`. You can guide the operator routing logic by providing a docstring inside each operation. Alternatively, you can also train it with some labelled examples like in `train_clf.csv`. This is recommended for accuracy.
4. This decision router would be saved by the operator in your desired location.
5. Going forward, you can just load this decision router using something like `operator = OnboardingOperator().load(<router_save_path>)`.
6. You can then pass user input to it using `response = operator(<query>)`.

### Examples
Onboarding Operator example

```
User input: who me? I am of age fifty nine, my friend.

Selected operation: setAge
Inferred arugments: {'age': '59'}

It is indicated to be the age of the user.
Age has been set. Age= 59
```

A Food Delivery Operator example

```
User input: I want 10l of milk.
Selected operation: order
Inferred arugments: {'item_name': 'milk', 'quantity': '10', 'unit': 'liters'}

It is indicated that the user wants to place an order.
Calling orders API with: item_name=milk, quantity=10, unit=liters

```
