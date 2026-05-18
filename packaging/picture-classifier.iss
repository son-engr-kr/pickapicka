; Inno Setup script for Picture Classifier.
;
; Wraps the PyInstaller onedir bundle (dist/picture-classifier/) into a
; Windows installer. Build from the project root:
;     iscc /DMyAppVersion=0.1.4 packaging\picture-classifier.iss
;
; RepoRoot defaults to "." (run from the project root); CI passes an absolute
; path so the [Files] and OutputDir paths resolve regardless of the cwd.

#define MyAppName "Picture Classifier"
#define MyAppExeName "picture-classifier.exe"
#define MyAppPublisher "Hyoungseo Son"
#define MyAppURL "https://github.com/son-engr-kr/picture-classifier"
#ifndef MyAppVersion
  #define MyAppVersion "0.0.0-dev"
#endif
#ifndef RepoRoot
  #define RepoRoot "."
#endif

[Setup]
; AppId uniquely identifies this app for upgrades/uninstall — keep it stable.
AppId={{D2C8F1A4-3B6E-4A9C-8E5F-1A7B2C3D4E5F}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
DefaultDirName={autopf}\Picture Classifier
DefaultGroupName=Picture Classifier
DisableProgramGroupPage=yes
OutputDir={#RepoRoot}\dist\installer
OutputBaseFilename=Picture-Classifier-{#MyAppVersion}-windows-x64
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
WizardStyle=modern
UninstallDisplayName={#MyAppName}
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#RepoRoot}\dist\picture-classifier\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\Picture Classifier"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall Picture Classifier"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Picture Classifier"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,Picture Classifier}"; Flags: nowait postinstall skipifsilent
