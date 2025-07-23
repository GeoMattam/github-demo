# Write-Output "`n==== PORTS RUNNING PYTHON, JAVA, NODE, STREAMLIT ===="

# # Keywords to match in process names
# $keywords = @("python", "java", "node", "streamlit", "dotnet", "php")

# # Get all listening ports
# $netstat = netstat -aon | Select-String "LISTENING"

# foreach ($line in $netstat) {
#     $parts = ($line -split "\s+") -ne ''
#     if ($parts.Count -ge 5) {
#         $proto        = $parts[0]
#         $localAddress = $parts[1]
#         $state        = $parts[-2]
#         $procId       = $parts[-1]  # ✅ renamed from $pid

#         # Get process name via tasklist
#         $task = tasklist /FI "PID eq $procId" /FO CSV | Select-String -NotMatch "Image Name"
#         if ($task) {
#             $columns = $task -replace '"' -split ","
#             $procName = $columns[0].ToLower()

#             if ($keywords -contains $procName -or $procName -like "python*" -or $procName -like "node*" -or $procName -like "java*") {
#                 $port = ($localAddress -split ":")[-1]
#                 Write-Output ("{0,-12} PID:{1,-7} Port:{2,-6} Addr:{3,-22} State:{4}" -f $procName, $procId, $port, $localAddress, $state)
#             }
#         }
#     }
# }

# Write-Output "`n==== DONE ===="

Write-Output "`n==== ONE ENTRY PER PYTHON/JAVA/NODE PROCESS ===="

$keywords = @("python", "java", "node", "streamlit", "dotnet", "php")
$seenPIDs = @{}
$netstat = netstat -aon | Select-String "LISTENING"

foreach ($line in $netstat) {
    $parts = ($line -split "\s+") -ne ''
    if ($parts.Count -ge 5) {
        $proto        = $parts[0]
        $localAddress = $parts[1]
        $state        = $parts[-2]
        $procId       = $parts[-1]

        # Skip if we've already shown this process
        if ($seenPIDs.ContainsKey($procId)) {
            continue
        }

        $task = tasklist /FI "PID eq $procId" /FO CSV | Select-String -NotMatch "Image Name"
        if ($task) {
            $columns = $task -replace '"' -split ","
            $procName = $columns[0].ToLower()

            if ($keywords -contains $procName -or $procName -like "python*" -or $procName -like "node*" -or $procName -like "java*") {
                $port = ($localAddress -split ":")[-1]
                Write-Output ("{0,-12} PID:{1,-7} Port:{2,-6} Addr:{3,-22} State:{4}" -f $procName, $procId, $port, $localAddress, $state)
                $seenPIDs[$procId] = $true  # Mark this PID as seen
            }
        }
    }
}

Write-Output "`n==== DONE ===="
