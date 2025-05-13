

import marimo

__generated_with = "0.13.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import asyncio
    return (asyncio,)


@app.cell
def _():
    from tqdm.notebook import tqdm, trange
    return tqdm, trange


@app.cell
async def _(asyncio, tqdm):
    # Set total length explicitly
    total_length = 10
    progress_bar = tqdm(total=total_length)

    for k in range(total_length):
        await asyncio.sleep(0.1)  # Simulate work
        progress_bar.update()
        progress_bar.write(s=f"test{k}")
    progress_bar.close()
    return


@app.cell
async def _(asyncio, mo):
    with mo.status.progress_bar(total=5) as bar:
        for i in range(5):
            await asyncio.sleep(0.2)
            bar.update()
        bar.clear()
    return


@app.cell
async def _(asyncio, mo):
    with mo.status.spinner(title="Loading...") as _spinner:
        await asyncio.sleep(0.2)
        _spinner.update("Almost done")
        await asyncio.sleep(0.2)
        _spinner.update("Done")
    return


@app.cell
async def _(asyncio, tqdm):
    for j in tqdm(range(5)):
        await asyncio.sleep(0.2)
    return


@app.cell
async def _(asyncio, trange):
    for l in trange(5):
        await asyncio.sleep(0.2)
    return


@app.cell
def _():
    from transformers import Trainer
    from transformers.utils.import_utils import is_in_notebook, is_in_marimo
    from transformers.utils.notebook import NotebookProgressCallback
    return is_in_marimo, is_in_notebook


@app.cell
def _(is_in_marimo):
    is_in_marimo()
    return


@app.cell
def _(is_in_notebook):
    is_in_notebook()
    return


if __name__ == "__main__":
    app.run()
