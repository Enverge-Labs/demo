import marimo

__generated_with = "0.13.11"
app = marimo.App()


@app.cell
def _():
    # Welcome to the Enverge
    return


@app.cell
def _():
    ## This is a blank Marimo notebook
    return


@app.cell
def _():
    import requests
    resp = requests.get("http://localhost:8000/notebook/api/usage")
    resp.json()
    return


if __name__ == "__main__":
    app.run()
