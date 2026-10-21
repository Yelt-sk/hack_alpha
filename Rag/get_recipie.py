def get_recipe(model_name: str, task: str)->str:
    recipe = ""

    with open("recipe_book/Prompt.txt", "r") as f:
        recipe += f.read()

    with open(f"recipe_book/{model_name}/{task}.txt", "r") as f:
            recipe += f.read()

    with open(f"recipe_book/{model_name}.txt", "r") as f:
        recipe += f.read()

    return recipe

def main():
     recipie = get_recipe("FLAML", "binary_classification")
     print(recipie)

if __name__ == "__main__":
     main()