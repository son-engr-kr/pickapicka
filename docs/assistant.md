# With an AI assistant

An assistant that runs commands, such as [Claude Code](https://claude.com/claude-code),
can work a shoot with you through the `pickapicka` command. It looks at the photos
(contact sheets of a scene, a photo at screen size, a crop at 100% to check focus
or eyes), picks and rejects, rates, tries looks and applies them, and exports.
Ask it the way you would ask a person: "go through Scene 3 and keep the best of
each burst", "this one is too dark", "make the picks warmer and export them for
Instagram".

With the app open on the project, every change shows in the window as it is
made, and the assistant knows which photo you are on: "this photo" is the one
under your cursor or open in the viewer.

## Setting it up

There is nothing to set up. As Pickapicka opens, if Claude Code is on the
computer, it writes the skill that tells Claude Code about the commands, into
`~/.claude/skills/pickapicka/`, and brings it up to date after each update.
Start Claude Code anywhere and ask. The skill names the command by its full
path, which for the app installed from the `.pkg` on a Mac is inside it
(`/Applications/Pickapicka.app/Contents/MacOS/pickapicka`), so it needs nothing
on PATH.

**Preferences → AI assistants** says whether Claude Code was found, and turning
it off takes the skill away. A `SKILL.md` you wrote there yourself is left as it
is. `pickapicka install-skill` writes it by hand, or into another folder with
`--dir`.

The Windows installer's app has no console, so its command prints nothing and
it writes no skill; on Windows, install with `uv tool` ([Install](install.md))
for the command line.

## The commands

Every command prints one JSON object, and `pickapicka COMMAND --help` says what
it does and what each field means. Without `-p`, a command works on the project
open in the app; `-p` takes a project's name, its folder, or the photo folder
it is kept in.

| | |
|---|---|
| `status` | the app, its project, and the scene and photo on screen |
| `projects`, `scenes`, `people` | what there is |
| `photos`, `photo REL` | the photos with their scores and decisions, chosen by scene, decision, rating, person; one photo's whole record |
| `sheet`, `view`, `thumb` | image files to look at: numbered contact sheets, one photo (as shot, edited, a 100% crop, or with a look tried on), thumbnails |
| `decide`, `mark` | pick, review, reject; stars and colour labels |
| `presets`, `apply` | the presets and the twenty looks; laying one on photos |
| `quick`, `sliders` | sliders moved on many photos at once, auto tone, level |
| `edit` | a whole edit written onto photos, or saved as a version |
| `export` | the picks (or more) written out, edits applied |
| `undo` | the last command-line change put back |

## How it reaches the project

The app holds an open project in memory and writes all of it back on each
change, so a command never writes the project's files while the app has it
open: it asks the app to make the change, exactly as a click in the window
would, and the window reloads the photos. With the app closed, or open on
another project, the command loads the project itself and saves it the same way.

Each change from the command line is remembered, twenty per project, and
`pickapicka undo` puts back the photos it changed as they were before it. An
export from the command line leaves your export settings as you last set them.
