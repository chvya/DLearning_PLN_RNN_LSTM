Set shell = CreateObject("WScript.Shell")
Set files = CreateObject("Scripting.FileSystemObject")
root = files.GetParentFolderName(WScript.ScriptFullName)
python = root & "\.venv\Scripts\pythonw.exe"
If Not files.FileExists(python) Then
    MsgBox "No se encontro .venv. Instala Python 3.12 y las dependencias de requirements-lock.txt.", 48, "Ancora"
    WScript.Quit 1
End If
shell.CurrentDirectory = root
shell.Run Chr(34) & python & Chr(34) & " " & Chr(34) & root & "\codigo\abrir_interfaz.py" & Chr(34), 0, False
