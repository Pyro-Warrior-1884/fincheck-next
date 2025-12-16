"use client"

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts"
import ChartWrapper from "../ChartWrapper"

export default function CpuTimeSeriesChart({
  data,
}: {
  data: any[]
}) {
  const series = data.map((d, i) => ({
    step: i + 1,
    cpu: d.cpu_percent ?? 0,
    model: d.model,
  }))

  return (
    <ChartWrapper
      title="CPU Utilization Time Series"
      description="CPU usage pattern across inference executions"
    >
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={series}>
          <XAxis dataKey="step" />
          <YAxis unit="%" />
          <Tooltip />
          <Line
            type="monotone"
            dataKey="cpu"
            strokeWidth={2}
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartWrapper>
  )
}
