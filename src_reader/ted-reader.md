### SimpleDirectoryReader

#### Supported file types

SimpleDirectoryReader is a general and great way to get started
In LlamaHub, There are lots of readers[Data Loaders](https://llamahub.ai/?tab=readers)


Binary files
```
.csv - comma-separated values
.docx - Microsoft Word
.epub - EPUB ebook format
.hwp - Hangul Word Processor
.ipynb - Jupyter Notebook
.jpeg, .jpg - JPEG image
.mbox - MBOX email archive
.md - Markdown
.mp3, .mp4 - audio and video
.pdf - Portable Document Format
.png - Portable Network Graphics
.ppt, .pptm, .pptx - Microsoft PowerPoint
```


Test files
```
JSON
Plain text
```


#### Usage

The most basic usage is to pass an input_dir which will load all supported files in the directory.

```
from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(input_dir="path/to/directory")
documents = reader.load_data()
```

for parallel processing

```
documents = reader.load_data(num_workers=4)
```

for reading from subdirectories

```
SimpleDirectoryReader(input_dir="path/to/directory", recursive=True)
```

Iterating over files as they load

```
reader = SimpleDirectoryReader(input_dir="path/to/directory", recursive=True)
all_docs = []

for docs in reader.iter_data():
    # do something with documents
    all_docs.extend(docs)

```

Restricting the files loaded

```
SimpleDirectoryReader(input_files=["path/to/files", "path/to/file2"])

or 

SimpleDirectoryReader(input_dir="path/to/directory", exclude=["path/to/file1", "path/to/file2"])

```

Only load files with extensions

```
SimpleDirectoryReader(input_dir="path/to/directory", required_exts=[".pdf", ".docx"])
```

To limit the number of files to be loaded

```
SimpleDirectoryReader(input_dir="path/to/directory", num_files_limit=100)
```

Specifying file encoding#
SimpleDirectoryReader expects files to be utf-8 encoded but you can override this using the encoding parameter:

```
SimpleDirectoryReader(input_dir="path/to/directory", encoding="latin-1")
```


#### Extracting metadata
SimpleDirectoryReader will automatically attach a metadata dictionary to each Document object. By default, this dictionary has these items:

file_path: the full filesystem path to the file, including the file name (string)
file_name: the file name, including suffix (string)
file_type: the MIME type of the file, as guessed by `mimetypes.guess_type() (string)
file_size: the size of the file, in bytes (integer)
creation_date, last_modified_date, last_accessed_date: the creation, modification, and access dates for the file, normalized to the UTC timezone. See Date and time metadata below (string)


```
def get_meta(file_path):
    return {"foo": "bar", "file_path": file_path}


reader = SimpleDirectoryReader(
    input_dir="path/to/directory", file_metadata=get_meta
)

docs = reader.load_data()
print(docs[0].metadata["foo"])  # prints "bar"
```


#### Support for External FileSystems

As with other modules, the SimpleDirectoryReader takes an optional fs parameter that can be used to traverse remote filesystems.

This can be any filesystem object that is implemented by the fsspec protocol. The fsspec protocol has open-source implementations for a variety of remote filesystems including AWS S3, Azure Blob & DataLake, Google Drive, SFTP, and many others.

Here's an example that connects to S3:

```
from s3fs import S3FileSystem

s3_fs = S3FileSystem(key="...", secret="...")
bucket_name = "my-document-bucket"

reader = SimpleDirectoryReader(
    input_dir=bucket_name,
    fs=s3_fs,
    recursive=True,  # recursively searches all subdirectories
)

documents = reader.load_data()
print(documents)
```


#### Extendint to other file types

You can extend SimpleDirectoryReader to read other file types by passing a dictionary of file extensions to instances of BaseReader as file_extractor. 
A BaseReader should read the file and return a list of Documents. For example, to add custom support for .myfile files :

```
from llama_index.core import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader
from llama_index.core import Document


class MyFileReader(BaseReader):
    def load_data(self, file, extra_info=None):
        with open(file, "r") as f:
            text = f.read()
        # load_data returns a list of Document objects
        return [Document(text=text + "Foobar", extra_info=extra_info or {})]


reader = SimpleDirectoryReader(
    input_dir="./data", file_extractor={".myfile": MyFileReader()}
)

documents = reader.load_data()
print(documents)
```