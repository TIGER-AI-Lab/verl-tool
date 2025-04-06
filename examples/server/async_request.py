from pyston import PystonClient, File
import asyncio

async def main():
    client = PystonClient(base_url="") # use your own url
    codes = ["print('Hello world')"] * 1000
    outputs = await asyncio.gather(*[asyncio.create_task(client.execute("python", [File(code)])) for code in codes])
    for output in outputs:
        print("output: ", output)
    
if __name__ == "__main__":
    asyncio.run(main())