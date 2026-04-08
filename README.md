LPBF Tool Handoff – Local Setup and Run Instructions
Step-by-step setup guide for local use on Windows and macOS

This project started from a real LPBF problem: important manufacturability and process-related issues are often discovered too late, after time has already been spent on design and preparation. In response, I developed an LPBF-aware topology optimization and decision-support workflow that brings those concerns earlier into the process.

1. Purpose
This document explains exactly how to download, open, set up, and run the LPBF-aware topology optimization tool on a local computer. The goal is to make the setup process simple, predictable, and easy to follow for a new user.
2. What You Will Receive
You will receive a Google Drive link that contains the full project handoff package.
The package should include the complete project folder with:
•	backend code
•	frontend UI
•	LPBF core package files
•	scripts and project utilities
•	artifacts and required runtime files
•	Docker setup files
•	setup documentation
Important: Do not try to run the project directly from Google Drive in the browser. Download it first, then extract it onto your computer.
3. Before You Start
This project is intended to run on a full computer, not on a phone or tablet.
Supported local platforms:
•	Windows
•	macOS
Recommended requirements:
•	A local desktop or laptop with enough free disk space
•	A modern web browser
•	Docker Desktop installed for the preferred setup method
•	Enough available memory for LPBF runs, especially larger ones
4. Download the Project from Google Drive
Windows
1.	Open the Google Drive link.
2.	Locate the shared project handoff file or folder.
3.	Click Download.
4.	Wait for the download to finish.
5.	Open your Downloads folder and confirm the file is there.
macOS
6.	Open the Google Drive link.
7.	Locate the shared project handoff file or folder.
8.	Click Download.
9.	Wait for the download to finish.
10.	Open Finder, then open the Downloads folder and confirm the file is there.
5. Extract or Open the Downloaded Project
If you received a compressed archive, extract it first.
Windows
11.	Go to the downloaded file in Downloads.
12.	Right-click the file.
13.	Click Extract All.
14.	Choose a location such as Desktop or Documents.
15.	Click Extract.
16.	Open the extracted project folder.
macOS
17.	Go to the downloaded file in Downloads.
18.	Double-click the archive.
19.	macOS will extract it automatically.
20.	Open the extracted project folder in Finder.
If you received a normal folder instead of an archive, move or copy it to a local location such as Desktop, Documents, or another project folder and open it there.
6. Confirm the Project Folder Looks Correct
After opening the project folder, confirm that you can see the main project contents. Important items should include:
•	backend
•	ui
•	lpbf_to
•	artifacts
•	scripts
•	Dockerfile.backend
•	Dockerfile.ui
•	docker-compose.yml
If these are missing, stop and request the full project handoff again before continuing.
7. Preferred Setup Method – Docker Desktop
Docker Desktop is the preferred way to run the project because it reduces environment and dependency issues across different machines.
7.1 Install Docker Desktop
Windows
21.	Open your web browser.
22.	Search for Docker Desktop.
23.	Download the Windows version.
24.	Install Docker Desktop.
25.	Open Docker Desktop after installation.
26.	Wait until Docker Desktop finishes starting.
macOS
27.	Open your web browser.
28.	Search for Docker Desktop.
29.	Download the correct version for your Mac.
30.	Install Docker Desktop.
31.	Open Docker Desktop.
32.	Wait until Docker Desktop finishes starting.
Do not continue until Docker Desktop is open and running.
7.2 Open a Terminal in the Project Folder
Windows
33.	Open the project folder in File Explorer.
34.	Click in the folder path bar.
35.	Type powershell.
36.	Press Enter.
This opens PowerShell inside the project folder.
macOS
37.	Open the project folder in Finder.
38.	Open Terminal.
39.	Type cd (with a space after it).
40.	Drag the project folder into the Terminal window.
41.	Press Enter.
This moves Terminal into the project folder.
7.3 Start the Project with Docker
In the terminal, run the following command:
docker compose up --build

This command builds and starts both the backend and frontend services. The first startup may take a little time.
8. Open the Tool in Your Browser
After Docker finishes starting the services, open your web browser.
Use these addresses:
•	Frontend UI: http://localhost:5173
•	Backend API/docs: http://localhost:8000/docs
If the UI opens successfully in the browser, the setup is working.
9. Basic Workflow Inside the Tool
42.	Open the UI in the browser.
43.	Define the design domain.
44.	Add supports and loads.
45.	Choose process settings.
46.	Start the optimization run.
47.	Monitor progress.
48.	Review results and analytics.
49.	Export outputs such as preview geometry, watertight STL, and summary artifacts.
10. Where Results Are Saved
Generated files are saved inside the project folder structure.
Important output areas are typically under:
•	artifacts
•	outputs
After a run completes, check those folders for generated artifacts and exported outputs.
11. How to Stop the Tool
If you started the project with Docker, return to the terminal and press Ctrl + C.
To stop and clean up the running containers more cleanly, run:
docker compose down

12. Manual Fallback Method (Only If Docker Is Not Used)
Use this method only if Docker Desktop is not available.
12.1 Start the Backend
Open a terminal in the project root folder and run:
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

Leave that terminal open.
12.2 Start the Frontend
Open a second terminal, move into the ui folder, and run:
yarn dev --host 0.0.0.0 --port 5173

Leave that terminal open as well.
12.3 Open the Browser
•	Frontend UI: http://localhost:5173
•	Backend API/docs: http://localhost:8000/docs
13. What to Expect the First Time
•	The first startup may take longer than later runs.
•	Docker build may take several minutes depending on the machine.
•	The project should be run from a full local folder, not from cloud preview.
•	Smaller validation runs are better for first-time testing.
•	Larger runs may need stronger hardware and more available memory.
14. Basic Troubleshooting
Problem: If Docker does not work:Make sure Docker Desktop is installed and already running.
Problem: If the frontend page does not open:Check whether the frontend service or frontend terminal is running.
Problem: If the backend docs page does not open:Check whether the backend service or backend terminal is running.
Problem: If the run feels heavy:Start with a smaller validation case first.
Problem: If extracted files look incomplete:Confirm that the project root still contains backend, ui, lpbf_to, artifacts, and docker-compose.yml.
15. Recommended First-Time Use
For the first setup, follow this exact order:
50.	Download the project from Google Drive.
51.	Extract it locally.
52.	Open Docker Desktop.
53.	Open a terminal in the project folder.
54.	Run docker compose up --build.
55.	Open http://localhost:5173.
56.	Open http://localhost:8000/docs.
