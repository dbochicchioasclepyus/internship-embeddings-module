const CodeEditor = ({ code }: { code: number[][] }) => {
  const codeString = code.map((line) => line.join(", ")).join("\n");

  return (
    <div className="code-editor">
      <pre className="code-token">{codeString}</pre>
    </div>
  );
};

export default CodeEditor;
