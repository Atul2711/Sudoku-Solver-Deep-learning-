from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
from numpy import argmax
import os
from tensorflow.keras.utils import load_img 
from tensorflow.keras.utils import get_file 

app = FastAPI()
origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware, 
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = methods,
    allow_headers = headers    
)

from image_prcoesses import extract as extract_img_grid
from digit_Recognition_CNN import run as create_and_save_Model
from predict import extract_number_image as sudoku_extracted
from solve import  ConstarintBacktracking as solve_sudoku


def display_gameboard(sudoku):
    for i in range(len(sudoku)):
        if i % 3 == 0:
            if i == 0:
                print(" ┎─────────┰─────────┰─────────┒")
            else:
                print(" ┠─────────╂─────────╂─────────┨")

        for j in range(len(sudoku[0])):
            if j % 3 == 0:
                print(" ┃ ", end=" ")

            if j == 8:
                print(sudoku[i][j] if sudoku[i][j] != 0 else ".", " ┃")
            else:
                print(sudoku[i][j] if sudoku[i][j] != 0 else ".", end=" ")

    print(" ┖─────────┸─────────┸─────────┚")

def main(img):
    # Calling the image_prcoesses.py extract function to get a processed np.array of cells
    image_grid = extract_img_grid(img)
    print("Image Grid extracted")

    # note we have alreday created and stored the model but if you want to do that again use the following command
    # create_and_save_Model()

    # Sudoku extract
    sudoku = sudoku_extracted(image_grid)
    print("Extracted and predict digits in the Sudoku")

    print("\n\nSudoku:")
    display_gameboard(sudoku)

    print("\nSolving the Sudoku...\n")
    solvable, solved = solve_sudoku(sudoku)

    if(solvable):
        print("\nSolved Sudoku:")
        display_gameboard(solved)

    print("Program End")



@app.get("/")
async def root():
    return {"message": "SUdoku Solver Vision API!"}


@app.post("/prediction")
async def get_net_image_prediction(image_link: str = ""):
    if image_link == "":
        return {"message": "No image link provided"}
    
    img_path = get_file(
        origin = image_link
    )
    img = load_img(
        img_path, 
        target_size = (224, 224)
    )
    main(img_path)

    return {
        "model-prediction": "hello",
    }

if __name__ == '__main__':

    port = int(os.environ.get('PORT', 5000))
    run(app, host="0.0.0.0", port=port)

# python -m uvicorn main:app --reload