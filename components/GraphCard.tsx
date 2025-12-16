export default function GraphCard({
  title,
  children,
}: {
  title: string
  children: React.ReactNode
}) {
  return (
    <div className="rounded-2xl bg-white p-5 shadow-sm border">
      <h3 className="mb-3 text-sm font-semibold text-gray-700">
        {title}
      </h3>
      {children}
    </div>
  )
}
