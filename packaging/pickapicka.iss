; Inno Setup script for Pickapicka.
;
; Wraps the PyInstaller onedir bundle (dist/pickapicka/) into a
; Windows installer. Build from the project root:
;     iscc /DMyAppVersion=0.1.4 packaging\pickapicka.iss
;
; RepoRoot defaults to "." (run from the project root); CI passes an absolute
; path so the [Files] and OutputDir paths resolve regardless of the cwd.

#define MyAppName "Pickapicka"
#define MyAppExeName "pickapicka.exe"
#define MyAppPublisher "Hyoungseo Son"
#define MyAppURL "https://github.com/son-engr-kr/pickapicka"
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
DefaultDirName={autopf}\Pickapicka
DefaultGroupName=Pickapicka
DisableProgramGroupPage=yes
OutputDir={#RepoRoot}\dist\installer
OutputBaseFilename=Pickapicka-{#MyAppVersion}-windows-x64
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
WizardStyle=modern
UninstallDisplayName={#MyAppName}
UninstallDisplayIcon={app}\{#MyAppExeName}
; The installer's own icon; the app's comes with the exe (pickapicka.spec).
SetupIconFile={#RepoRoot}\packaging\icons\pickapicka.ico
; Installs from before the rename (Picture Classifier, 0.9.0 and earlier) have
; this AppId too, so they upgrade in place: into the folder they already use,
; with the Start menu entries moved to the new name.
UsePreviousGroup=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[InstallDelete]
; What an install from before the rename leaves that nothing below replaces.
Type: files; Name: "{app}\picture-classifier.exe"
Type: files; Name: "{autodesktop}\Picture Classifier.lnk"
Type: files; Name: "{autoprograms}\Picture Classifier\Picture Classifier.lnk"
Type: files; Name: "{autoprograms}\Picture Classifier\Uninstall Picture Classifier.lnk"
Type: dirifempty; Name: "{autoprograms}\Picture Classifier"

[Files]
Source: "{#RepoRoot}\dist\pickapicka\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\Pickapicka"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall Pickapicka"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Pickapicka"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,Pickapicka}"; Flags: nowait postinstall skipifsilent
