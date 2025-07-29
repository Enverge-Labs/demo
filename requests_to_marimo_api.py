import marimo

__generated_with = "0.14.7"
app = marimo.App()


@app.cell
def _():
    # Welcome to the Enverge
    return


@app.cell
def _():
    import requests
    resp = requests.get("http://localhost:8000/notebook/api/usage")
    resp.json()
    return


if __name__ == "__main__":
    app.run()
