using System;
using System.ComponentModel;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows.Forms;

// 将界面、浏览器渲染进程和判题子进程放在同一个 Windows Job 中。
// 窗口正常退出或异常退出后，关闭 Job 句柄都会清理其余进程。
internal static class Launcher
{
    [StructLayout(LayoutKind.Sequential)]
    private struct StartupInfo
    {
        public uint cb;
        public IntPtr reserved, desktop, title;
        public uint x, y, width, height, charsX, charsY, fill, flags;
        public ushort show, reservedBytes;
        public IntPtr reserved2, stdin, stdout, stderr;
    }
    [StructLayout(LayoutKind.Sequential)]
    private struct ProcessInfo { public IntPtr process, thread; public uint processId, threadId; }
    [StructLayout(LayoutKind.Sequential)]
    private struct BasicLimits
    {
        public long perProcessTime, perJobTime;
        public uint flags;
        public UIntPtr minWorkingSet, maxWorkingSet;
        public uint activeProcessLimit;
        public UIntPtr affinity;
        public uint priorityClass, schedulingClass;
    }
    [StructLayout(LayoutKind.Sequential)]
    private struct IoCounters { public ulong readOps, writeOps, otherOps, readBytes, writeBytes, otherBytes; }
    [StructLayout(LayoutKind.Sequential)]
    private struct ExtendedLimits
    {
        public BasicLimits basic;
        public IoCounters io;
        public UIntPtr processMemory, jobMemory, peakProcessMemory, peakJobMemory;
    }
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr CreateJobObject(IntPtr attributes, string name);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetInformationJobObject(IntPtr job, int infoClass, ref ExtendedLimits info, uint length);
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool CreateProcess(string application, StringBuilder command, IntPtr processAttributes,
        IntPtr threadAttributes, bool inherit, uint flags, IntPtr environment, string directory,
        ref StartupInfo startup, out ProcessInfo process);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern uint ResumeThread(IntPtr thread);
    [DllImport("kernel32.dll")]
    private static extern uint WaitForSingleObject(IntPtr handle, uint milliseconds);
    [DllImport("kernel32.dll")]
    private static extern bool GetExitCodeProcess(IntPtr process, out uint code);
    [DllImport("kernel32.dll")]
    private static extern bool TerminateProcess(IntPtr process, uint code);
    [DllImport("kernel32.dll")]
    private static extern bool CloseHandle(IntPtr handle);

    private static string Quote(string argument)
    {
        var output = new StringBuilder("\"");
        int slashes = 0;
        foreach (char c in argument)
        {
            if (c == '\\') { slashes++; continue; }
            if (c == '"') output.Append('\\', slashes * 2 + 1);
            else output.Append('\\', slashes);
            output.Append(c);
            slashes = 0;
        }
        output.Append('\\', slashes * 2);
        return output.Append('"').ToString();
    }

    [STAThread]
    private static int Main(string[] args)
    {
        bool nonInteractive = Array.IndexOf(args, "--smoke-test") >= 0;
        IntPtr job = IntPtr.Zero;
        ProcessInfo process = new ProcessInfo();
        try
        {
            string root = AppDomain.CurrentDomain.BaseDirectory;
            string python = Path.Combine(root, "runtime", "pythonw.exe");
            string entry = Path.Combine(root, "app", "desktop.py");
            var command = new StringBuilder(Quote(python) + " " + Quote(entry));
            foreach (string arg in args) command.Append(" " + Quote(arg));

            job = CreateJobObject(IntPtr.Zero, null);
            if (job == IntPtr.Zero) throw new Win32Exception();
            var limits = new ExtendedLimits();
            limits.basic.flags = 0x2000; // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if (!SetInformationJobObject(job, 9, ref limits, (uint)Marshal.SizeOf(limits)))
                throw new Win32Exception();

            var startup = new StartupInfo();
            startup.cb = (uint)Marshal.SizeOf(startup);
            // 暂停启动，先加入 Job，再允许运行，避免子进程逃出生命周期管理。
            if (!CreateProcess(python, command, IntPtr.Zero, IntPtr.Zero, false,
                0x08000004, IntPtr.Zero, root, ref startup, out process))
                throw new Win32Exception();
            if (!AssignProcessToJobObject(job, process.process))
            {
                TerminateProcess(process.process, 1);
                throw new Win32Exception();
            }
            ResumeThread(process.thread);
            WaitForSingleObject(process.process, 0xffffffff);
            uint exitCode;
            GetExitCodeProcess(process.process, out exitCode);
            if (exitCode != 0 && !nonInteractive)
                MessageBox.Show("启动失败。请查看 %LOCALAPPDATA%\\PyTorchInterviewLab\\desktop.log。", "手撕实验室");
            return (int)exitCode;
        }
        catch (Exception error)
        {
            if (!nonInteractive) MessageBox.Show("无法启动手撕实验室：\n" + error.Message + "\n请保持 exe、app 和 runtime 文件夹在一起。", "手撕实验室");
            return 1;
        }
        finally
        {
            if (process.thread != IntPtr.Zero) CloseHandle(process.thread);
            if (process.process != IntPtr.Zero) CloseHandle(process.process);
            if (job != IntPtr.Zero) CloseHandle(job);
        }
    }
}
