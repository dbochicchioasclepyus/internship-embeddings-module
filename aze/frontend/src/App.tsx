import React, { useEffect, useState } from "react";
import "./index.css";
import CodeEditor from "./components/CodeEditor";
type DataProps = { id: number; name: string };

function App() {
  const [data, setData] = useState<DataProps[]>([]); // State variable to store fetched data
  const [result, setResult] = useState<{ message: string; data: any }>(); // State variable to store fetched data
  const [selectedOptions, setSelectedOptions] = useState<string[]>([]); // State variable to store selected options
  const [loading, setLoading] = useState(false);
  const [name, setName] = useState("");
  const [selectedLabel, setSelectedLabel] = useState<string>(""); // State variable to store the selected label

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
        setData(data);
        setSelectedOptions([]);
      } else {
        console.error("Error:", response.statusText);
      }
    } catch (error) {
      console.error("Error:", error);
    }
  };

  const toggleCheckbox = (value: string) => {
    if (selectedOptions.includes(value)) {
      setSelectedOptions(selectedOptions.filter((item) => item !== value));
    } else {
      setSelectedOptions([...selectedOptions, value]);
    }
  };

  const handleLabelChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedLabel(event.target.value);
  };

  const postData = async () => {
    setLoading(true);
    setResult({ message: "", data: [] });
    try {
      setName(selectedOptions.join(","));
      const requestBody = JSON.stringify(selectedOptions);

      const response = await fetch(
        "http://localhost:7071/api/HttpTrigger?action=embed",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json", // Specify the content type as JSON
          },
          body: requestBody,
        }
      );

      if (response.ok) {
        setLoading(false);
        const data = await response.json();
        console.log(data);

        setName(data.name);
        setResult(data);
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
        data: result?.data,
        name,
        label: selectedLabel,
      };
      console.log(requestBody);

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
    setResult({ message: "", data: [] });
    setSelectedOptions([]);
    setLoading(false);
    setName("");
  };

  return (
    <div className="App">
      <div className="left">
        <h1>Embeddings</h1>
        <div className="checkboxdiv">
          {data.length > 0 ? (
            data.map((item, index) => (
              <div className="checkbox" key={index}>
                <input
                  type="checkbox"
                  value={item.name}
                  checked={selectedOptions.includes(item.name)}
                  onChange={() => toggleCheckbox(item.name)}
                />
                {item.name}
              </div>
            ))
          ) : (
            <span>No files available</span>
          )}
        </div>
        <div className="buttons">
          <button onClick={postData}>Embed</button>
          <button
            onClick={uploadData}
            style={result?.data.length ? undefined : { cursor: "no-drop" }}
            disabled={result?.data.length ? false : true}
          >
            Upload to Zilliz
          </button>
          <button onClick={onReset}>Reset</button>
        </div>
      </div>
      <div className="right">
        <div className="result">
          {result?.data.length > 0 ? (
            <CodeEditor code={result?.data} />
          ) : (
            "No data available"
          )}
          {loading ? <span>Loading. It may take a while...</span> : undefined}
        </div>
        <div className="labelDropdown">
          <select value={selectedLabel} onChange={handleLabelChange}>
            <option value="SpeechContext">Speech Context</option>
            <option value="EventContext">Event Context</option>
            <option value="Speaker">Speaker</option>
          </select>
        </div>
      </div>
    </div>
  );
}

export default App;
