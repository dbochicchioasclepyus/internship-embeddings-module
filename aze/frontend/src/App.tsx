import { useEffect, useState } from "react";
import "./index.css";
import CodeEditor from "./components/CodeEditor";
type DataProps = { id: number; name: string };
type PostProps = {
  message: string;
  embeddings: any;
  name: string;
  content: string;
};
function App() {
  const [fileNames, setFileNames] = useState<DataProps[]>([]);
  const [result, setResult] = useState<PostProps>();
  const [selectedOptions, setSelectedOptions] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [name, setName] = useState("");
  const [label, setLabel] = useState<string>("");

  useEffect(() => {
    getData();
  }, []);

  const getData = async () => {
    try {
      const response = await fetch(
        "http://localhost:7071/api/HttpTrigger?action=list"
      );
      if (response.ok) {
        const data: DataProps[] = await response.json();
        setFileNames(data);
        setSelectedOptions([]);
      } else {
        console.error("Error:", response.statusText);
      }
    } catch (error) {
      console.error("Error:", error);
    }
  };

  const handleCheckboxChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const value = event.target.value;
    setSelectedOptions((prev) =>
      prev.includes(value)
        ? prev.filter((opt) => opt !== value)
        : [...prev, value]
    );
  };

  const postData = async () => {
    setLoading(true);
    try {
      const response = await fetch(
        "http://localhost:7071/api/HttpTrigger?action=embed",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(selectedOptions),
        }
      );

      if (response.ok) {
        setLoading(false);
        const data = await response.json();
        setResult(data);
        setName(data.name.join(","));
      } else {
        setLoading(false);
        console.error("Error:", response.statusText);
      }
    } catch (error) {
      setLoading(false);
      console.error("Error:", error);
    }
  };

  const uploadData = async () => {
    try {
      const requestBody = {
        embeddings: result?.embeddings,
        name,
        label,
        content: result?.content,
      };

      const response = await fetch(
        "http://localhost:7071/api/HttpTrigger?action=upload",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(requestBody),
        }
      );

      if (response.ok) {
        const data = await response.json();
        console.log(data);
        alert("Uploaded");
      } else {
        console.log(await response.json());
      }
    } catch (error) {
      console.error("Error:", error);
    }
  };

  const onReset = () => {
    setResult({ message: "", embeddings: [], name: "", content: "" });
    setSelectedOptions([]);
    setLoading(false);
    setName("");
  };

  return (
    <div className="App">
      <div className="left">
        <h1>Embeddings</h1>
        <div className="checkboxdiv">
          {fileNames.length > 0 ? (
            fileNames.map((item, index) => (
              <div className="checkbox" key={index}>
                <input
                  type="checkbox"
                  value={item.name}
                  checked={selectedOptions.includes(item.name)}
                  onChange={handleCheckboxChange}
                />
                {item.name}
              </div>
            ))
          ) : (
            <span>No files available</span>
          )}
        </div>
        <div className="buttons">
          <button
            disabled={selectedOptions.length > 0 ? false : true}
            onClick={postData}
          >
            Embed
          </button>
          <button
            onClick={uploadData}
            style={
              result?.embeddings.length ? undefined : { cursor: "no-drop" }
            }
            disabled={result?.embeddings.length ? false : true}
          >
            Upload to Zilliz
          </button>
          <button onClick={onReset}>Reset</button>
        </div>
      </div>
      <div className="right">
        <div className="result">
          {result?.embeddings.length > 0 ? (
            <CodeEditor code={result?.embeddings} />
          ) : (
            "No data available"
          )}
          {loading ? <span>Loading. It may take a while...</span> : undefined}
        </div>
        <div>
          <select
            className="custom-select"
            onChange={(e) => setLabel(e.target.value)}
            value={label}
          >
            <option value="EventContext">EventContext</option>
            <option value="SpeechContext">SpeechContext</option>
            <option value="Speaker">Speaker</option>
          </select>
        </div>
      </div>
    </div>
  );
}

export default App;
